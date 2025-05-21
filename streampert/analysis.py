from gala.units import UnitSystem
from astropy import units as u
usys = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian)
import jax.numpy as jnp

import jax
jax.config.update("jax_enable_x64", True)

from streamsculptor import potential
from streamsculptor import JaxCoords as jc
import diffrax

import streamsculptor as ssc

import interpax
import tqdm
import numpy as np
from streamsculptor import perturbative as pert
from streamsculptor.GenerateImpactParams import ImpactGenerator

from streamsculptor.subhalostatistics import RateCalculator as RC
from streamsculptor.perturbedstream import gen_perturbed_stream
from streamsculptor.perturbative import BaseStreamModelChen25 
import os
from scipy.stats import binned_statistic
import inferencetools as it
from functools import partial
import equinox as eqx
import streampert as sp

print('backend -- ' + str(jax.devices()))

velconv = (1*u.kpc/u.Myr).to(u.km/u.s).value



@jax.jit
def icrs_to_gd1(ra_rad, dec_rad):
    """
    define a *differentiable* coordinate transfrom from ra and dec --> gd1 phi1, phi2
    Using the rotation matrix from Koposov+2010
    ra_rad: icrs ra [radians]
    dec_red: icrs dec [radians]
    """
    R = jnp.array(
        [
            [-0.4776303088, -0.1738432154, 0.8611897727],
            [0.510844589, -0.8524449229, 0.111245042],
            [0.7147776536, 0.4930681392, 0.4959603976],
        ]
    )


    icrs_vec = jnp.vstack([jnp.cos(ra_rad)*jnp.cos(dec_rad),
                           jnp.sin(ra_rad)*jnp.cos(dec_rad),
                           jnp.sin(dec_rad)]).T

    stream_frame_vec = jnp.einsum('ij,kj->ki',R,icrs_vec)
    
    phi1 = jnp.arctan2(stream_frame_vec[:,1],stream_frame_vec[:,0])*180/jnp.pi
    phi2 = jnp.arcsin(stream_frame_vec[:,2])*180/jnp.pi

    
    return phi1, phi2

@jax.jit
def get_phi12_from_stream(stream):
    """
    Differentiable helper function that takes us from simulated stream --> phi1, phi2
    """
    ra_s, dec_s, dist_ = jax.vmap(jc.simcart_to_icrs)(stream[:,:3])
    phi1_model, phi2_model = icrs_to_gd1(ra_s*jnp.pi/180, dec_s*jnp.pi/180)
    return phi1_model, phi2_model

def load_and_concatenate(path, min_iter, max_iter):
    pert_out_list = []
    r_s_root_list = []
    
    for i in range(min_iter, max_iter + 1):
        file_path = os.path.join(path, f'{i}.npy')
        data = np.load(file_path, allow_pickle=True).item()
        pert_out_list.append(data['pert_out'][1])
        r_s_root_list.append(data['r_s_root'])
        
    combined_pert_out = jnp.concatenate(pert_out_list, axis=1)
    combined_r_s_root = jnp.concatenate(r_s_root_list, axis=0)
    
    return combined_pert_out, combined_r_s_root, data['pert_out'][0]

@jax.jit
def compute_dispersion(x: jnp.array,  mask: jnp.array):
    count = jnp.sum(mask)
    mean = jnp.sum(x * mask) / count
    sq_diff = ( (x - mean) ** 2 ) * mask
    variance = jnp.sum(sq_diff) / (count - 1)
    return jnp.sqrt(variance)


def run_inference(  
                    perturbation_path: str,
                    prog_path: str,
                    save_path: str,
                    gd1_data_path: str,
                    key: jax.random.PRNGKey,
                    constrain_norm: bool,
                    constrain_conc_fac: bool,
                    constrain_M_hm: bool,
                    num_samps_norm: int,
                    num_samps_conc_fac: int,
                    num_samps_M_hm: int,
                    t_age: float,
                    Msat: float,
                    r_s = 0.02,
                    t_dissolve = -200.0,
                    norm_default = 1.0,
                    conc_fac_default = 1.0,
                    M_hm_default = 0.0,
                    norm_min = 0.5,
                    norm_max = 3.5,
                    log10M_min = 5.,
                    log10M_max = 9.,
                    conc_fac_min = 0.4,
                    conc_fac_max = 1.4,
                    M_hm_min = 1e-3,
                    M_hm_max = 1e9,
                    N_sim_per_rel= 1000,
                    n_vmap = 4,
                    phi1_min = -80,
                    phi1_max = 15.,
                    phi2_min = -8.,
                    phi2_max = 5.,
                    min_iter_load = 0,
                    max_iter_load = 19,
                    phi1_min_length = -150.,
                    phi1_max_length = 55.,
                    num_bins = 80,

                    ):

    data_gd1 = np.load(gd1_data_path,allow_pickle=True).item()
    phi1_bins = data_gd1['bins']
    median_sig = data_gd1['median_sid']
    median_sig_err = data_gd1['median_err']

    phi1_bins_jax = jnp.asarray(phi1_bins)
    data_stds_jax = jnp.asarray(median_sig)
    data_errs_jax = jnp.asarray(median_sig_err)

    prog_wtoday = jnp.load(prog_path,allow_pickle=True).item()
    pos = prog_wtoday.pos.xyz.to(u.kpc).value
    vel = prog_wtoday.vel.d_xyz.to(u.kpc/u.Myr).value
    prog_wtoday = jnp.hstack([pos,vel])

    pot = potential.GalaMilkyWayPotential(units=usys)
    IC = pot.integrate_orbit(w0=prog_wtoday, t0=0.0, t1=-t_age,ts=jnp.array([-t_age])).ys[0]

    

    # Load perturbation data
    combined_pert_out, combined_r_s_root, unpert = load_and_concatenate(perturbation_path, min_iter_load, max_iter_load) 
    

    phi1_u, phi2_u = get_phi12_from_stream(unpert)
    inside = ( phi1_u>phi1_min_length ) & (phi1_u < phi1_max_length)

    # Compute unperturbed rv track
    _,_, rv_unpert = jax.vmap(jc.simvel_to_ICRS)(unpert[:,:3], unpert[:,3:])
    bins_left =   jnp.linspace(-110,-25,40)
    bins_right = jnp.linspace(-16,40,40)

    H_left = binned_statistic(phi1_u, rv_unpert, statistic=jnp.nanmean,bins=bins_left)
    H_right = binned_statistic(phi1_u, rv_unpert, statistic=jnp.nanmean,bins=bins_right)
    centers = jnp.hstack([0.5*(H_left.bin_edges[1:]+H_left.bin_edges[:-1]), 
                        0.5*(H_right.bin_edges[1:]+H_right.bin_edges[:-1])])
    means = jnp.hstack([H_left.statistic, H_right.statistic])
    rv_spl = interpax.Interpolator1D(x=centers, f=means, method='cubic2')





    length = ssc.compute_stream_length(stream=unpert[inside], phi1=phi1_u[inside])
    orb_fwd = pot.integrate_orbit(w0=IC, t0=-t_age, t1=0.0, ts=jnp.linspace(-t_age,0,4_000))

    ratecalc = RC(orbit=orb_fwd,
        t_age=t_age,
        b_max_fac=10.0,
        l_obs=length
        )
    
    trial_samples = ratecalc.sample_masses(log10M_min=log10M_min,
                               log10M_max=log10M_max,
                               key=jax.random.PRNGKey(222412),
                               normalization=norm_max,
                               array_length=20)
    N_encounter_rate = trial_samples['N_encounter_rate']
    array_length = N_encounter_rate*1.1

    @jax.jit
    def get_mask(stream, phi1_min=phi1_min, phi1_max=phi1_max, phi2_min=phi2_min, phi2_max=phi2_max):
        p1, p2 = get_phi12_from_stream(stream)
        inside = (p1 > phi1_min) & (p1 < phi1_max) & (p2 > phi2_min) & (p2 < phi2_max)
        return inside#, p1, p2

    
    @jax.jit
    def gen_real_and_measure(key: jax.random.PRNGKey, 
                            combined_pert_out : jnp.ndarray, 
                            combined_r_s_root: jnp.ndarray,
                            array_length: int,
                            conc_fac = 1.0, 
                            norm = 1.0,
                            M_hm = 0.0,
                            log10M_min = 5.,
                            log10M_max = 9.,
                            ):
        """
        Warning: must specify phi1_bins_jax and rv_spl and mask_func outside of this function
                as global variables.
        """
        keys = jax.random.split(key,3)
        samps = ratecalc.sample_masses(log10M_min=log10M_min,
                                log10M_max=log10M_max,
                                key=keys[0],
                                normalization=norm,
                                M_hm=M_hm,
                                array_length=1_000)
    
        out = sp.gen_stream_realization(unpert, 
                                combined_pert_out, 
                                combined_r_s_root,
                                jnp.where(samps['log10_mass']==0,0,10**samps['log10_mass']), 
                                keys[1], 
                                concentration_fac=conc_fac,
                                use_binned_idx_finder = True,
                                num_bins = num_bins)
        
        stream = out[0]

        mask_out = get_mask(stream)
        phi1_p, phi2_p = get_phi12_from_stream(stream)
        std_phi2  = compute_dispersion(phi2_p,mask_out)

        
        disp, counts = it.compute_binned_dispersion(
                    stream = stream,
                    phi1_bins=phi1_bins_jax,
                    mask_func=get_mask,
                    rv_spl=rv_spl)
        disp = disp * velconv

        
        return disp, std_phi2, counts
        

    
    mapped_func = jax.vmap(gen_real_and_measure, in_axes=(0, None, None, None, None, None, None, None, None))
    mapped_func  = jax.jit(mapped_func)

    if constrain_norm & constrain_conc_fac:
        log10_norm = jnp.linspace(jnp.log10(norm_min), jnp.log10(norm_max), num_samps_norm)
        log10_conc_fac = jnp.linspace(jnp.log10(conc_fac_min), jnp.log10(conc_fac_max), num_samps_conc_fac)

        log10N, log10C = jnp.meshgrid(log10_norm, log10_conc_fac)
        params_stack = jnp.vstack([log10N.flatten(),log10C.flatten()]).T

        def inference_loop(
                params_stack: jnp.array,
                key: jax.random.PRNGKey,
                #array_length: int,
                ):

            
            keys = jax.random.split(key, len(params_stack))
            samples_arr = np.zeros((len(params_stack), int(jnp.ceil(N_sim_per_rel/n_vmap)), n_vmap, 5))
            std_arr = np.zeros((len(params_stack), int(jnp.ceil(N_sim_per_rel/n_vmap)), n_vmap))
            counts_arr =  np.zeros((len(params_stack), int(jnp.ceil(N_sim_per_rel/n_vmap)), n_vmap, 5))

            
            
            
            

            for i in tqdm.tqdm(range(len(params_stack))):
                log10N, log10C = params_stack[i]
                N = 10**log10N
                conc_fac = 10**log10C
                
                keys_batch = jax.random.split(keys[i],samples_arr.shape[1]*samples_arr.shape[2])
                keys_vmap = keys_batch.reshape(samples_arr.shape[1],samples_arr.shape[2],2)
                
                for j in range(len(keys_vmap)):
                    samps, std_phi2, counts = mapped_func(keys_vmap[j], 
                                                combined_pert_out, 
                                                combined_r_s_root,
                                                array_length,
                                                conc_fac,
                                                N,
                                                M_hm_default,
                                                log10M_min,
                                                log10M_max
                                                )
                  
                    
                    samples_arr[i,j,:,:]  = samps
                    std_arr[i,j,:] = std_phi2
                    counts_arr[i,j,:,:] = counts
                    
            return samples_arr, std_arr, counts_arr

        output = inference_loop(params_stack = params_stack, 
                                key = key
                                )
                                #array_length = array_length)


    if constrain_M_hm & constrain_conc_fac:
        log10_conc_fac = jnp.linspace(jnp.log10(conc_fac_min), jnp.log10(conc_fac_max), num_samps_conc_fac)
        log10_M_hm = jnp.linspace(jnp.log10(M_hm_min), jnp.log10(M_hm_max), num_samps_M_hm)

        log10C, log10M = jnp.meshgrid(log10_conc_fac, log10_M_hm)
        params_stack = jnp.vstack([log10C.flatten(),log10M.flatten()]).T

        def inference_loop(
                params_stack: jnp.array,
                key: jax.random.PRNGKey,
                array_length: int,
                ):

            
            keys = jax.random.split(key, len(params_stack))
            samples_arr = np.zeros((len(params_stack), int(jnp.ceil(N_sim_per_rel/n_vmap)), n_vmap, 5))
            std_arr = np.zeros((len(params_stack), int(jnp.ceil(N_sim_per_rel/n_vmap)), n_vmap))
            counts_arr =  np.zeros((len(params_stack), int(jnp.ceil(N_sim_per_rel/n_vmap)), n_vmap, 5))

        

            for i in tqdm.tqdm(range(len(params_stack))):
                log10C, log10M = params_stack[i]
                conc_fac = 10**log10C
                M_hm = 10**log10M
                
                keys_batch = jax.random.split(keys[i],samples_arr.shape[1]*samples_arr.shape[2])
                keys_vmap = keys_batch.reshape(samples_arr.shape[1],samples_arr.shape[2],2)
                
                for j in range(len(keys_vmap)):
                    samps, std_phi2, counts = mapped_func(keys_vmap[j], 
                                                combined_pert_out, 
                                                combined_r_s_root,
                                                array_length,
                                                conc_fac,
                                                norm_default,
                                                M_hm,
                                                log10M_min,
                                                log10M_max,
                                                )

                                                                    
                    samples_arr[i,j,:,:]  = samps
                    std_arr[i,j,:] = std_phi2
                    counts_arr[i,j,:,:] = counts
                    
            return samples_arr, std_arr, counts_arr

        output = inference_loop(params_stack = params_stack, 
                                key = key,
                                array_length = array_length)

    if constrain_norm & constrain_M_hm:
        log10_norm = jnp.linspace(jnp.log10(norm_min), jnp.log10(norm_max), num_samps_norm)
        log10_M_hm = jnp.linspace(jnp.log10(M_hm_min), jnp.log10(M_hm_max), num_samps_M_hm)

        log10N, log10M = jnp.meshgrid(log10_norm, log10_M_hm)
        params_stack = jnp.vstack([log10N.flatten(),log10M.flatten()]).T

        def inference_loop(
                params_stack: jnp.array,
                key: jax.random.PRNGKey,
                array_length: int,
                ):

            
            keys = jax.random.split(key, len(params_stack))
            samples_arr = np.zeros((len(params_stack), int(jnp.ceil(N_sim_per_rel/n_vmap)), n_vmap, 5))
            std_arr = np.zeros((len(params_stack), int(jnp.ceil(N_sim_per_rel/n_vmap)), n_vmap))
            counts_arr =  np.zeros((len(params_stack), int(jnp.ceil(N_sim_per_rel/n_vmap)), n_vmap, 5))
            
            
            for i in tqdm.tqdm(range(len(params_stack))):
                log10N, log10M = params_stack[i]
                N = 10**log10N
                M_hm = 10**log10M
                
                keys_batch = jax.random.split(keys[i],samples_arr.shape[1]*samples_arr.shape[2])
                keys_vmap = keys_batch.reshape(samples_arr.shape[1],samples_arr.shape[2],2)
                
                for j in range(len(keys_vmap)):
                    samps, std_phi2, counts = mapped_func(keys_vmap[j], 
                                                combined_pert_out, 
                                                combined_r_s_root,
                                                array_length,
                                                conc_fac_default,
                                                N,
                                                M_hm,
                                                log10M_min,
                                                log10M_max,
                                                )

                    samples_arr[i,j,:,:]  = samps
                    std_arr[i,j,:] = std_phi2
                    counts_arr[i,j,:,:] = counts
                    
            return samples_arr, std_arr, counts_arr

        output = inference_loop(params_stack = params_stack, 
                                key = key,
                                array_length = array_length)

    save_dict =  dict(output=output, params_stack=params_stack)
    print('saving to ' + save_path)
    np.save(save_path, save_dict)
    return None

    
