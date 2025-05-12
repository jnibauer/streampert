from gala.units import UnitSystem
from astropy import units as u
usys = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian)
import jax.numpy as jnp
from astropy.coordinates import SkyCoord, Galactocentric


import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)

from streamsculptor import potential
from streamsculptor import JaxCoords as jc
import diffrax


from streamsculptor import JaxCoords as jc


import streamsculptor as ssc
from streamsculptor import JaxCoords as jc

from streamsculptor import perturbative as pert


@jax.jit
def compute_binned_dispersion(phi1_bins: jnp.array, phi1: jnp.array, drv: jnp.array, model_mask: jnp.array):
    """
    Compute the standard deviation (dispersion) of `drv` values within each bin defined by `phi1_bins`.

    For each bin, this function computes the sample standard deviation of the `drv` values whose corresponding
    `phi1` values fall into that bin. Bins are defined by `phi1_bins`, and the result is computed using 
    vectorized mapping with `jax.vmap`.

    Parameters
    ----------
    phi1_bins : array_like
        Array of bin edges. Must be 1-dimensional and sorted in ascending order.
        The number of bins is `len(phi1_bins) - 1`.
    phi1 : array_like
        Array of `phi1` values to be binned. Must be the same shape as `drv`.
    drv : array_like
        Data values for which to compute dispersion. Must be the same shape as `phi1`.
    model_mask : array_like
        Boolean mask array of the same shape as `phi1` and `drv`. Only values where mask is True are considered.

    Returns
    -------
    dispersions : ndarray
        Array of standard deviations (one per bin). If a bin has fewer than 2 entries,
        its value is `NaN`.
    counts : ndarray
        Number of elements in each bin.

    Notes
    -----
    This function is JIT-compiled with `jax.jit` and uses `jax.vmap` to parallelize over bins.
    """
    bin_indices = jnp.digitize(phi1, phi1_bins) - 1  # bins are 0-indexed
    num_bins = len(phi1_bins) - 1
    def compute_for_bin(i):
        mask = (bin_indices == i) & model_mask 
        count = jnp.sum(mask)
        mean = jnp.sum(drv * mask) / count
        sq_diff = (drv - mean) ** 2 * mask
        variance = jnp.sum(sq_diff) / (count - 1)
        return jnp.where(count > 1, jnp.sqrt(variance), jnp.nan), count
    
    return jax.vmap(compute_for_bin)(jnp.arange(num_bins))



@jax.jit
def find_idx_from_mass(masses: jnp.array, r_s_values: jnp.array, key: jax.random.PRNGKey, concentration_fac = 1.0,):
    """
    Finds the indices in `r_s_values` that are closest to the expected scale radii
    computed from input `masses`, using a greedy matching strategy that avoids repeated indices.

    The matching ensures that each mass is assigned a unique radius index in `r_s_values`
    that is closest to its expected scale radius.

    Parameters:
    -----------
    masses : jax.numpy.ndarray, shape (N,)
        Array of subhalo masses for which to estimate scale radii.
    r_s_values : jax.numpy.ndarray, shape (M,)
        Array of sampled scale radius values (e.g., from a grid or simulation output).
        Assumes M >= N for unique matching.     
    key : jax.random.PRNGKey 
        Random number key for shuffling input masses, such that one set of masses gives different
        orbits for different keys. 
    concentration_fac : float
        Multiplicative factor applied to scale radius to mimic different subhalo concentrations.
        For CDM-like behavior, use 1.0; use <1.0 for more concentrated subhalos.

    Returns:
    --------
    jax.numpy.ndarray, shape (N,)
        Array of indices into `r_s_values`, one per input mass, such that each index is unique
        and corresponds to the closest available radius to the expected value.

    Notes:
    ------
    - This function uses a greedy algorithm to avoid repeated indices, but it is not globally optimal.
    - It is fully JAX-compatible and JIT-compilable.
    - Intended for use in batched simulations or derivative sampling workflows.
    """
    keys = jax.random.split(key, 2)
    masses = jax.random.permutation(key=keys[0], x=masses)
    # Added noise to concentration to select for different orbits randomly. We wiill
    # correct for differences from input scale-radius using perturbation theory
    concentration_noise = jax.random.uniform(keys[1], minval=0.8, maxval=1.8, shape=(len(masses),))
    expected_r_s = 1.05 * jnp.sqrt(masses / 1e8) * concentration_fac * concentration_noise
    N, M = expected_r_s.shape[0], r_s_values.shape[0]

    # Compute pairwise distances (N x M)
    dists = jnp.abs(expected_r_s[:, None] - r_s_values[None, :])

    # Mask to mark used indices in r_s_values
    def body_fun(i, val):
        dists, used, result = val
        # Mask out used indices by setting their distance to a large number
        masked_dists = dists.at[:, used].set(jnp.inf)
        idx = jnp.argmin(masked_dists[i])
        used = used.at[i].set(idx)
        result = result.at[i].set(idx)
        return (dists, used, result)

    result = jnp.full(N, -1)
    used = jnp.full(N, -1)

    _, _, result = jax.lax.fori_loop(0, N, body_fun, (dists, used, result))
    return result
    

@jax.jit
def gen_stream_realization(unpert: jnp.ndarray, 
                           derivs: jnp.ndarray, 
                           r_s_root: jnp.ndarray, 
                           m_arr: jnp.ndarray, 
                           key: jax.random.PRNGKey, 
                           concentration_fac = 1.0):
    """
    Generates a realization of a perturbed stream given a mass realization and a set of root scale-radii
    for the derivatives. Adjusting `concentration_fac` will not adjust the orbits.

    Parameters
    ----------
    unpert : jnp.ndarray
        Array representing the unperturbed stream.
    derivs : jnp.ndarray
        Array of derivatives used to compute the perturbations.
    r_s_root : jnp.ndarray
        Root scale-radii for the derivatives, which serve as a reference for the perturbations.
    m_arr : jnp.ndarray
        Array of masses used to calculate the stream realization.
    key : jax.random.PRNGKey
        PRNG key used for random number generation.
    concentration_fac : float, optional
        Adjustment factor for the concentration of the root scale-radius. Default is 1.0.

    Returns
    -------
    slin : jnp.ndarray
        Perturbed stream realization.
    idx_take : jnp.ndarray
        Indices used to select elements from the `r_s_root` and derivatives arrays.

    Notes
    -----
    The function computes the perturbations by selecting scale-radii (`r_s_vals`) based on the mass 
    realization and the `concentration_fac`. It calculates the deviations (`drs`) from the expected 
    scale-radius _before_ applying `concentraction_fac` and uses these to compute the perturbed stream 
    via the input derivatives.
    """
    idx_take = find_idx_from_mass(m_arr, r_s_root, key, 1.0)
    
    r_s_vals = r_s_root.at[idx_take].get()
    expected_r_s = 1.05 * jnp.sqrt(m_arr / 1e8) * concentration_fac
    drs = expected_r_s - r_s_vals
  
    derivs_m = derivs.at[:,idx_take,:6].get()
    derivs_rs = derivs.at[:,idx_take,6:].get()
    
    m_arr =  m_arr[None,:,None]
    drs_arr = drs[None,:,None]
    
    derivs = jnp.sum( derivs_m * m_arr, axis=1) + jnp.sum(derivs_rs * m_arr * drs_arr, axis=1)
    slin = unpert + derivs
    return slin, idx_take








