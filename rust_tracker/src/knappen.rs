use std::{cmp::Ordering, f64::consts::PI, io::Error};
// use rayon::prelude::*;
// use std::sync::Mutex;
// use ndarray::{Array1, Array2, ArrayBase, Data, Ix1};
use nalgebra::{Matrix4, Vector2};
use ode_solvers::{Rk4, SVector, System, Vector3, Vector4, Dop853}; //, dop853::Dop853
use roots::{find_root_brent, SimpleConvergency};

const ETA_ETA_MATRIX: Matrix4<f64> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0,
);

const INV_GEV_IN_SEC: f64 = 1.0 / (1.52e24);
const INV_GEV_IN_CM: f64 = 1.9732697880000002e-14;

const EPSILON: f64 = 1e6;
const INV_GEV_TO_NANO_SEC: f64 = 1e9 / (1.52e24);

pub static ATLAS_RADII_PIXEL: [f64; 8] = [3.10, 5.05, 8.85, 12.25, 29.9, 37.1, 44.3, 51.4];
const ATLAS_Z_RANGE: [f64; 8] = [32.0, 40.0, 40.0, 40.0, 74.9, 74.9, 74.9, 74.9];

// static LENGTH_IN_PHI: [f64; 8] = [0.005, 0.005, 0.005, 0.005, 0.0034, 0.0034, 0.0034, 0.0034];

static PHI_SIZE: [f64; 8] = [
    0.0016129,
    0.000990099,
    0.000564972,
    0.000408163,
    0.000113712,
    0.0000916442,
    0.0000767494,
    0.0000661479,
];

static Z_SIZE: [f64; 8] = [0.025, 0.04, 0.04, 0.04, 0.116, 0.116, 0.116, 0.116];

const NA: f64 = 6.022e23;
const ME: f64 = 0.511e-3;
const RE: f64 = 2.81794033e-15 * 100.0; // fm to m conversion and then to cm
const EV2GEV: f64 = 10e-9;

// static HBAR_OMEGA_P_SI: f64 = 31.05;
static STERNHEIMER_A_SI: f64 = 0.1492;
static STERNHEIMER_K_SI: f64 = 3.2546;
static STERNHEIMER_X0_SI: f64 = 0.2015;
static STERNHEIMER_X1_SI: f64 = 2.8716;
static STERNHEIMER_I_SI: f64 = 173.0;
static STERNHEIMER_CBAR_SI: f64 = 4.4355;
static STERNHEIMER_DELTA0_SI: f64 = 0.14;

static Z_SI: i32 = 14;

static A_SI: f64 = 28.085;

static RHO_SI: f64 = 2.33;

// static X0_SI: f64 = 9.37;

// static LAMBDA_SI: f64 = 46.52;

// static LAMBDAC_SI: f64 = 30.16;

// static EEHOLE_SI: f64 = 3.6;

const THICKNESSES: [f64; 8] = [0.023, 0.025, 0.025, 0.025, 0.0285, 0.0285, 0.0285, 0.0285];
const DELTA_R: [f64; 8] = THICKNESSES;

fn beta(gamma: f64) -> f64 {
    (1.0 - 1.0 / gamma.powi(2)).sqrt()
}

fn eta(gamma: f64) -> f64 {
    (gamma.powi(2) - 1.0).sqrt()
}

// fn tmax(ee: f64, mm: f64, me: f64) -> f64 {
//     let eta_ee_mm = eta(ee / mm);
//     let beta_ee_mm = beta(ee / mm);
//     2.0 * me * eta_ee_mm.powi(2)
//         / (1.0 + 2.0 * eta_ee_mm / beta_ee_mm * me / mm + (me / mm).powi(2))
// }

fn delta(eta: f64) -> f64 {
    let x0 = STERNHEIMER_X0_SI;
    let x1 = STERNHEIMER_X1_SI;
    let cbar = STERNHEIMER_CBAR_SI;
    let a = STERNHEIMER_A_SI;
    let k = STERNHEIMER_K_SI;
    let delta0 = STERNHEIMER_DELTA0_SI;

    let log_eta = eta.log10();
    let out: f64;
    if log_eta >= x1 {
        out = 2.0 * std::f64::consts::LN_10 * log_eta - cbar
    } else if log_eta >= x0 {
        out = 2.0 * std::f64::consts::LN_10 * log_eta - cbar + a * (x1 - log_eta).powf(k)
    } else {
        out = delta0 * 10f64.powf(2.0 * (log_eta - x0))
    }
    // println!("{:}", out);
    out
}

fn delta_p_pdg(x: f64, gamma: f64, z: f64) -> f64 {
    // TODO: Ensure this function works as expected!
    

    // 1.0 * Rho_Si * 4 * np.pi * NA * re**2 * me * z**2 *0.5 * Z_Si * x / (A_Si * beta(gamma)**2) * 
    //              (np.log(2 * me * eta(gamma)**2 / (eV2GeV * SternheimerI_Si)) + 
    //               np.log((1.0 * Rho_Si * 4 * np.pi * NA * re**2 * me * z**2) * 
    //                      0.5 * Z_Si * x / (A_Si * beta(gamma)**2) / (eV2GeV * SternheimerI_Si)) - 
    //              beta(gamma)**2 - Delta(eta(gamma), Si) + 0.2)
    

    let output: f64 = 1.0 * RHO_SI * 4.0 * PI * NA * RE.powi(2) * ME * z.powi(2) * 0.5 * (Z_SI as f64) * x / 
                 (A_SI * beta(gamma).powi(2)) * (
                 (2.0 * ME * eta(gamma).powi(2) / (EV2GEV * STERNHEIMER_I_SI)).ln() + 
                 ((1.0 * RHO_SI * 4.0 * PI * NA * RE.powi(2) * ME * z.powi(2) * 
                   0.5 * (Z_SI as f64) * x / (A_SI * beta(gamma).powi(2)) / 
                   (EV2GEV * STERNHEIMER_I_SI)).ln()) - 
                 beta(gamma).powi(2) - delta(eta(gamma)) + 0.2
                 );

    // println!("{:}", output);
    output

}

fn de(gamma: f64, delta_r: f64) -> f64 {
    1e3 * delta_p_pdg(delta_r, gamma, 1.0)
}

fn s(x1: &Vector3<f64>, x2: &Vector3<f64>) -> Vector3<f64> {
    (x1 - x2).normalize()
}

fn vpar(v: &Vector3<f64>, x1: &Vector3<f64>, x2: &Vector3<f64>) -> Vector3<f64> {
    let s_dir = s(x1, x2);
    s_dir * v.dot(&s_dir)
}

fn vper(v: &Vector3<f64>, x1: &Vector3<f64>, x2: &Vector3<f64>) -> Vector3<f64> {
    v - vpar(v, x1, x2)
}

fn gammasc(v: f64) -> f64 {
    1.0 / (1.0 - v.powi(2)).sqrt()
}

fn gamma(v: &Vector3<f64>) -> f64 {
    1.0 / (1.0 - v.norm_squared()).sqrt()
}

fn lorentz(p4: &Vector4<f64>) -> Matrix4<f64> {
    let p4_1_4: Vector3<f64> = Vector3::new(p4.y, p4.z, p4.w);
    let v: f64 = p4_1_4.norm() / p4.x;
    let n: Vector3<f64> = p4_1_4 / p4_1_4.norm();
    let gscv = gammasc(v);

    Matrix4::new(
        gscv, -gscv * v * n.x, -gscv * v * n.y, -gscv * v * n.z,
        -gscv * v * n.x, 1.0 + (gscv - 1.0) * n.x.powi(2), (gscv - 1.0) * n.x * n.y, (gscv - 1.0) * n.x * n.z,
        -gscv * v * n.y, (gscv - 1.0) * n.y * n.x, 1.0 + (gscv - 1.0) * n.y.powi(2), (gscv - 1.0) * n.y * n.z,
        -gscv * v * n.z, (gscv - 1.0) * n.z * n.x, (gscv - 1.0) * n.z * n.y, 1.0 + (gscv - 1.0) * n.z.powi(2),
    )
}


fn yzw(p4: &Vector4<f64>) -> Vector3<f64> {
    Vector3::new(p4.y, p4.z, p4.w)
}

fn dp_dt(
    v: &Vector3<f64>,
    x1: &Vector3<f64>,
    x2: &Vector3<f64>,
    q: f64,
    ecm: &Vector3<f64>,
    bcm: &Vector3<f64>,
    t: f64,
    quirkflag: bool,
) -> Vector3<f64> {
    if quirkflag {
        let vp = vper(v, x1, x2);
        let vp_dot_vp = vp.dot(&vp);
        let s_x1_x2 = s(x1, x2);
        let vpar_vec = vpar(v, x1, x2);
        let vp_div_sqrt = vp / (1.0 - vp_dot_vp).sqrt();
        -t * ((1.0 - vp_dot_vp).sqrt() * s_x1_x2 + vpar_vec.dot(&s_x1_x2) * vp_div_sqrt)
            + q * (ecm + v.cross(bcm))
    } else {
        q * (ecm + v.cross(bcm))
    }
}

struct ParticleSystem {
    m: f64,
    bcm: Vector3<f64>,
    t: f64,
    quirkflag: bool,
}

type State = ode_solvers::SVector<f64, 15>;
type Time = f64;

impl System<State> for ParticleSystem {
    fn system(&self, _t: Time, _y: &State, _dy: &mut State) {
        let gammabeta1 = _y.fixed_rows::<3>(0).into_owned();
        let gammabeta2 = _y.fixed_rows::<3>(3).into_owned();
        let x1 = _y.fixed_rows::<3>(6).into_owned();
        let x2 = _y.fixed_rows::<3>(9).into_owned();
        let ef = _y.fixed_rows::<3>(12).into_owned();

        let dgammabeta1dt = (1.0 / self.m)
            * dp_dt(
                &(gammabeta1 / (1.0 + gammabeta1.norm_squared()).sqrt()),
                &x1,
                &x2,
                1.0,
                &ef,
                &self.bcm,
                self.t,
                self.quirkflag,
            );
        let dgammabeta2dt = (1.0 / self.m)
            * dp_dt(
                &(gammabeta2 / (1.0 + gammabeta2.norm_squared()).sqrt()),
                &x2,
                &x1,
                -1.0,
                &ef,
                &self.bcm,
                self.t,
                self.quirkflag,
            );

        let dx1dt = gammabeta1 / (1.0 + gammabeta1.norm_squared()).sqrt();
        let dx2dt = gammabeta2 / (1.0 + gammabeta2.norm_squared()).sqrt();

        _dy.fixed_rows_mut::<3>(0).copy_from(&dgammabeta1dt);
        _dy.fixed_rows_mut::<3>(3).copy_from(&dgammabeta2dt);
        _dy.fixed_rows_mut::<3>(6).copy_from(&dx1dt);
        _dy.fixed_rows_mut::<3>(9).copy_from(&dx2dt);
        _dy.fixed_rows_mut::<3>(12).fill(0.0); // dEfdt is zero
    }
}

fn find_tracks(
    vec1: &Vector4<f64>,
    vec2: &Vector4<f64>,
    root_sigma: f64,
    quirkflag: bool,
    b: &Vector3<f64>,
) -> Result<(Interpolator, Interpolator), Error> {
    let added: Vector4<f64> = vec1 + vec2;

    let m: f64 = (vec1.dot(&(ETA_ETA_MATRIX * vec1))).sqrt();

    let t: f64 = (root_sigma * 1e-9).powi(2);
    let boost: Matrix4<f64> = lorentz(&added);
    let boost_back: Matrix4<f64> = lorentz(&(ETA_ETA_MATRIX * added));

    for i in 0..4 {
        for j in 0..4 {
            if !boost[(i, j)].is_finite() || !boost_back[(i, j)].is_finite() || !m.is_finite() || !t.is_finite() {
                return Err(Error::new(std::io::ErrorKind::Other, "NaN in FindTracks"));
            }
        }
    }

    let vcm: Vector3<f64> = yzw(&added) / added.x;

    let gamma_vcm: f64 = gamma(&vcm);

    let ecm: Vector3<f64> = gamma_vcm * vcm.cross(b);
    let bcm: Vector3<f64> =
        gamma_vcm * b - ((gamma_vcm - 1.0) * (b.dot(&vcm) / vcm.norm()) * (vcm / vcm.norm()));

    // let mut tmax = false;

    // let e: f64 = (4.0 * std::f64::consts::PI * 1.0 / 137.0).sqrt();

    let system = ParticleSystem {
        m,
        bcm,
        t,
        quirkflag,
    };

    let mut y0: State = State::zeros();

    let gammabeta1_0: Vector3<f64> = (1.0 / m) * yzw(&(boost * vec1));
    let gammabeta2_0: Vector3<f64> = (1.0 / m) * yzw(&(boost * vec2));
    let x1_0: Vector3<f64> = Vector3::new(0.0, 10.0_f64.powi(-5), 0.0);
    let x2_0: Vector3<f64> = Vector3::new(0.0, -10.0_f64.powi(-5), 0.0);
    let ef_0: Vector3<f64> = ecm;

    // Set initial conditions in y0
    y0.fixed_rows_mut::<3>(0).copy_from(&gammabeta1_0);
    y0.fixed_rows_mut::<3>(3).copy_from(&gammabeta2_0);
    y0.fixed_rows_mut::<3>(6).copy_from(&x1_0);
    y0.fixed_rows_mut::<3>(9).copy_from(&x2_0);
    y0.fixed_rows_mut::<3>(12).copy_from(&ef_0);

    let total_time = 1e19;
    let num_steps = 10000000;
    let step_size = total_time / (num_steps as f64); // Approximate step size


    let event_fn: Box<dyn Fn(Time, &State, &Matrix4<f64>) -> bool> = Box::new(
        |time, state, boost_back| {
            let mult_vec: Vector4<f64> = Vector4::new(INV_GEV_IN_SEC, INV_GEV_IN_CM, INV_GEV_IN_CM, INV_GEV_IN_CM);
            let sumx12: Vector3<f64> = state.fixed_rows::<3>(6).into_owned() + state.fixed_rows::<3>(9).into_owned();
            let norm: f64 = (yzw(&mult_vec.component_mul(&(boost_back * Vector4::new(time, sumx12.x, sumx12.y, sumx12.z))))).norm();
            // println!("norm: {:}", norm);
            norm > 60.0
        },
    );

    // let mut stepper = Dop853::new(system, 0.1, 10e19, step_size, y0, 1e-5, 1e12, boost_back).with_event_fn(event_fn);

    let mut stepper = Rk4::new(system, 0.1, y0, total_time, step_size, boost_back).with_event_fn(event_fn);



    let res = stepper.integrate();


    // Handle result
    let mut sol0_values: Vec<Vector4<f64>> = Vec::new();
    let mut sol1_values: Vec<Vector4<f64>> = Vec::new();

    match res {
        Ok(_stats) => {
            let time_points: &Vec<f64> = stepper.x_out();
            let sol_states: &Vec<SVector<f64, 15>> = stepper.y_out();

            let conversion_vec: Vector4<f64> = Vector4::new(INV_GEV_IN_SEC, INV_GEV_IN_CM, INV_GEV_IN_CM, INV_GEV_IN_CM);
            for i in 0..time_points.len() {
                let sol0: Vector4<f64> = conversion_vec.component_mul(&(boost_back * Vector4::new(time_points[i], sol_states[i][6], sol_states[i][7], sol_states[i][8])));
                let sol1: Vector4<f64> = conversion_vec.component_mul(&(boost_back * Vector4::new(time_points[i], sol_states[i][9], sol_states[i][10], sol_states[i][11])));
                sol0_values.push(sol0);
                sol1_values.push(sol1);
            }

            return Ok((
                Interpolator::new(time_points.to_owned(), sol0_values.to_owned()),
                Interpolator::new(time_points.to_owned(), sol1_values.to_owned()),
            ));
        }
        Err(_) => {
            println!("An error occured.");
            Err(Error::new(std::io::ErrorKind::Other, "An error occured integrating in FindTracks."))
        }
    }
}

fn find_edges(
    vec1: &Vector4<f64>,
    vec2: &Vector4<f64>,
    root_sigma: f64,
    layer_number: i64,
) -> Vec<f64> {
    let added: Vector4<f64> = vec1 + vec2;

    let vcm: Vector4<f64> = added / 2.0;
    let vcm_inv: Vector4<f64> = Vector4::new(vcm.x, -vcm.y, -vcm.z, -vcm.w);
    let lvcm: Matrix4<f64> = lorentz(&vcm);
    let vecs_cm: Vector2<Vector4<f64>> = Vector2::new(lvcm * vec1, lvcm * vec2);

    let m: f64 = vecs_cm.x.dot(&(ETA_ETA_MATRIX * vecs_cm.x)).sqrt();
    let sigma: f64 = (root_sigma * 1e-9).powi(2);
    let v0: f64 = Vector3::new(vecs_cm.x.y, vecs_cm.x.z, vecs_cm.x.w).norm() / vecs_cm.x.x;

    let t_max: f64 = (m * v0) / ((1.0 - v0.powi(2)).sqrt() * sigma);
    let r_max: f64 = (m * (-1.0 + 1.0 / (1.0 - v0.powi(2)).sqrt())) / sigma;

    let endpoints: Vec<Vector4<f64>> = vecs_cm
        .iter()
        .map(|v: &Vector4<f64>| {
            let yzw = yzw(&v);
            let norm: f64 = yzw.norm();
            if norm != 0.0 {
                let unit_vector: Vector3<f64> = yzw / norm;
                Vector4::new(
                    t_max,
                    (unit_vector * r_max).x,
                    (unit_vector * r_max).y,
                    (unit_vector * r_max).z,
                )
            } else {
                Vector4::new(t_max, 0.0, 0.0, 0.0)
            }
        })
        .collect();

    let endpointslab: Vec<Vector4<f64>> = endpoints
        .iter()
        .map(|e: &Vector4<f64>| lorentz(&vcm_inv) * e)
        .collect();

    let el0: Vector4<f64> = endpointslab[0];
    let el1: Vector4<f64> = endpointslab[1];

    let el0_yzw: Vector3<f64> = yzw(&el0);
    let el1_yzw: Vector3<f64> = yzw(&el1);

    let vcm_yzw = yzw(&vcm);

    let a = vcm_yzw.xy().norm_squared() * INV_GEV_IN_CM.powi(2);

    let b_0 = 2.0 * vcm_yzw.xy().dot(&el0_yzw.xy()) * INV_GEV_IN_CM.powi(2);
    let c_0 = el0_yzw.xy().norm_squared() * INV_GEV_IN_CM.powi(2)
        - ATLAS_RADII_PIXEL[(layer_number - 1) as usize].powi(2);

    let b_1 = 2.0 * vcm_yzw.xy().dot(&el1_yzw.xy()) * INV_GEV_IN_CM.powi(2);
    let c_1 = el1_yzw.xy().norm_squared() * INV_GEV_IN_CM.powi(2)
        - ATLAS_RADII_PIXEL[(layer_number - 1) as usize].powi(2);

    let roots0: Vec<f64>;
    let roots1: Vec<f64>;
    if b_0.powi(2) - 4.0 * a * c_0 < 0.0 {
        // roots0 = vec![-b_0 / (2.0 * a)];
        roots0 = vec![];
    } else {
        let rad: f64 = (b_0.powi(2) - 4.0 * a * c_0).sqrt();
        roots0 = vec![(-b_0 + rad) / (2.0 * a), (-b_0 - rad) / (2.0 * a)];
    }
    if b_1.powi(2) - 4.0 * a * c_1 < 0.0 {
        // roots1 = vec![-b_1 / (2.0 * a)];
        roots1 = vec![];
    } else {
        let rad: f64 = (b_1.powi(2) - 4.0 * a * c_1).sqrt();
        roots1 = vec![(-b_1 + rad) / (2.0 * a), (-b_1 - rad) / (2.0 * a)];
    }

    let mut solutions: Vec<f64> = Vec::new();

    for root in roots0.iter() {
        solutions.push((vcm_yzw * *root + el0_yzw).z * INV_GEV_IN_CM);
    }
    for root in roots1.iter() {
        solutions.push((vcm_yzw * *root + el1_yzw).z * INV_GEV_IN_CM);
    }

    if solutions.iter().any(|&sol| sol.is_finite()) {
        return solutions;
    }

    return vec![0.0];
}

struct Interpolator {
    t: Vec<f64>,
    y: Vec<Vector4<f64>>,
}

impl Interpolator {
    fn new(t: Vec<f64>, y: Vec<Vector4<f64>>) -> Self {
        Interpolator { t, y }
    }

    fn interpolate(&self, t_value: f64) -> Vector4<f64> {
        let n = self.t.len();
        if n == 0 {
            return Vector4::new(0.0, 0.0, 0.0, 0.0);
        }
        if n == 1 {
            return self.y[0];
        }

        if t_value <= self.t[0] {
            // Linear extrapolation using the first two points
            let slope = (self.y[1] - self.y[0]) / (self.t[1] - self.t[0]);
            return self.y[0] + slope * (t_value - self.t[0]);
        }

        if t_value >= self.t[n - 1] {
            // Linear extrapolation using the last two points
            let slope = (self.y[n - 1] - self.y[n - 2]) / (self.t[n - 1] - self.t[n - 2]);
            return self.y[n - 1] + slope * (t_value - self.t[n - 1]);
        }

        for i in 0..n - 1 {
            if self.t[i] <= t_value && t_value <= self.t[i + 1] {
                let t0 = self.t[i];
                let t1 = self.t[i + 1];
                let y0 = &self.y[i];
                let y1 = &self.y[i + 1];
                let factor = (t_value - t0) / (t1 - t0);
                return y0 * (1.0 - factor) + y1 * factor;
            }
        }

        // Fallback, should not reach here if t is sorted
        self.y[n - 1]
    }
}

fn beta_beta(traj_map: &Interpolator, t0: f64) -> f64 {
    // return ((np.linalg.norm(trajMap([t0 + epsilon])[0][1:4]) - np.linalg.norm(trajMap([t0])[0][1:4])) / (trajMap([t0 + epsilon])[0][0] - trajMap([t0])[0][0])) / (3 * 10**10)

    let traj_map_t0_plus_epsilon: Vector4<f64> = traj_map.interpolate(t0 + EPSILON);
    let traj_map_t0: Vector4<f64> = traj_map.interpolate(t0);

    // return ((np.linalg.norm(trajMap([t0 + epsilon])[0][1:4]) - np.linalg.norm(trajMap([t0])[0][1:4])) / (trajMap([t0 + epsilon])[0][0] - trajMap([t0])[0][0])) / (3 * 10**10)

    (yzw(&traj_map_t0_plus_epsilon).norm() - yzw(&traj_map_t0).norm())
        / (traj_map_t0_plus_epsilon.x - traj_map_t0.x)
        / 3.0e10
}

fn gamma_gamma(traj_map: &Interpolator, t0: f64) -> f64 {
    return 1.0 / (1.0 - beta_beta(traj_map, t0).powi(2)).sqrt();
}

fn find_intersections(traj: &Interpolator) -> Vec<(i32, f64, f64, f64, f64, f64)> {
    let t_values = traj.t.clone();
    let r_values: Vec<f64> = t_values
        .iter()
        .map(|&t| {
            let traj_t = traj.interpolate(t);
            Vector3::new(traj_t[1], traj_t[2], 0.0).norm()
        })
        .collect();

    let coarse_list: Vec<(f64, f64, f64, f64)> = t_values
        .iter()
        .zip(&r_values)
        .zip(t_values.iter().skip(1).zip(&r_values).skip(1))
        .map(|((t0, r0), (t1, r1))| (*t0, *r0, *t1, *r1))
        .collect();

    let mut final_list = Vec::new();

    for layer in 0..8 {
        let layer_radius = ATLAS_RADII_PIXEL[layer];
        let phi_size_layer = PHI_SIZE[layer];
        let z_size_layer = Z_SIZE[layer];

        let layer_list: Vec<(f64, f64, f64, f64)> = coarse_list
            .iter()
            .filter(|row| (row.1 - layer_radius) * (row.3 - layer_radius) < 0.0)
            .cloned()
            .collect();

        for (t_low, _, t_high, _) in layer_list {
            let root = find_root_brent(
                t_low,
                t_high,
                |t| {
                    let traj_t = traj.interpolate(t);
                    Vector3::new(traj_t[1], traj_t[2], 0.0).norm() - layer_radius
                },
                &mut SimpleConvergency {
                    eps: 1e-12,
                    max_iter: 1000,
                },
            );

            if let Ok(t) = root {
                let traj_t = traj.interpolate(t);
                let z = traj_t[3];
                let phi = traj_t[2].atan2(traj_t[1]);
                let gamma = gamma_gamma(&traj, t);
                let beta = beta_beta(&traj, t);

                final_list.push((
                    (layer + 1) as i32,
                    (z / Z_SIZE[1]).round() * Z_SIZE[1],
                    (phi / phi_size_layer).round() * phi_size_layer,
                    de(gamma, DELTA_R[layer]),
                    beta.abs() * gamma,
                    INV_GEV_TO_NANO_SEC * gamma * t,
                ));
            }
        }
    }

    final_list
}

pub fn run_point(
    vec1: &Vector4<f64>,
    vec2: &Vector4<f64>,
    root_sigma: f64,
    plotflag: bool,
    quirkflag: bool,
) -> Result<Vec<(i32, f64, f64, f64, f64, f64, i32)>, Error> {
    let find_edges1: Vec<f64> = find_edges(vec1, vec2, root_sigma, 1);
    let find_edges2: Vec<f64> = find_edges(vec1, vec2, root_sigma, 2);

    if (!find_edges1.is_empty()
        && find_edges1
            .iter()
            .map(|x| x.abs())
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap()
            < ATLAS_Z_RANGE[0])
        || (!find_edges2.is_empty()
            && find_edges2
                .iter()
                .map(|x| x.abs())
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .unwrap()
                < ATLAS_Z_RANGE[1])
    {
        match find_tracks(vec1, vec2, root_sigma, quirkflag, &Vector3::new(0.0, 0.0, 1.18314e-16)) {
            Ok((sol_1, sol2)) => {
                if plotflag {}

                let intersections1 = find_intersections(&sol_1);
                let intersections2 = find_intersections(&sol2);
                // Rest of the code...
                let mut combined_intersections: Vec<(i32, f64, f64, f64, f64, f64, i32)> = intersections1
            .iter()
            .map(|&(a, b, c, d, e, f)| (a, b, c, d, e, f, 1))
            .chain(
                intersections2
                    .iter()
                    .map(|&(a, b, c, d, e, f)| (a, b, c, d, e, f, 2)),
            )
            .collect();

        combined_intersections
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        return Ok(combined_intersections);
            }
            Err(err) => {
                // Handle the error case if needed
                println!("Error: {:?}", err);
                // Return an appropriate value or propagate the error
                return Err(err);
            }
        }

        
    }
    return Err(Error::new(std::io::ErrorKind::Other, "An error occured in run_point."));
}

// fn main() {
//     let vec1: Vector4<f64> = Vector4::new(421.69956147, 258.12146064, 154.10248991, -254.86516886);
//     let vec2: Vector4<f64> = Vector4::new(202.65928421, -123.22431566, 12.442253162, 56.848428683);

//     let aa = run_point(&vec1, &vec2, 500.0, false, true);
//     println!("{:?}", aa);
// }


// py: [[1, -2.64, 0.5419344, 0.06288493374079132, 2.2561222483873955, 0.23785147553579533, 1], [1, 1.24, 2.9661231, 0.3252579692077869, 0.4233489829076971, 0.28755476030329946, 2], [2, -4.28, 0.5445544499999999, 0.07094490768551115, 2.019079012223044, 0.35946622884292295, 1], [2, -3.4, 1.703960379, 0.21505072615936255, 0.580332978007614, 0.9603263510333062, 2], [3, -8.08, 1.1180795879999998, 0.08476583530132205, 1.3749953127245347, 1.7521922842691802, 2], [3, -7.5600000000000005, 0.5519776439999999, 0.07937789075685336, 1.5376324051510992, 0.5339713479056355, 1], [4, -11.32, 0.948570812, 0.07320856273517314, 1.8421345372354019, 2.4194304893668277, 2], [4, -10.52, 0.564489429, 0.10347708958808463, 1.061347829459936, 0.6250279006577989, 1], [5, -27.6, 0.872284752, 0.08164090790911607, 2.0242909914352656, 5.623013728263143, 1], [5, -27.6, 1.033414656, 0.16400673162581364, 0.7863141720853037, 4.020151526463462, 2], [5, -27.400000000000002, 0.810993984, 1.0400952375646249, 0.2524648989256872, 2.2870108821910717, 2], [5, -27.080000000000002, 0.742766784, 0.615823116038202, 0.3355667616516371, 2.0086326745042995, 2], [6, -34.36, 0.9444851252, 0.0895650641125025, 1.600575235485153, 6.5564502771683815, 2], [6, -34.0, 0.8090349976000001, 0.09470028311951517, 1.445132821914007, 4.816050032683682, 1], [7, -41.0, 0.9854622959999999, 0.15213678408382755, 0.8377343057599682, 5.7369291366121695, 1], [7, -40.92, 0.8764781479999999, 0.08192699721197517, 2.0018760862522487, 8.320147153219061, 2], [7, -40.68, 0.8261305416, 2.2451760523455238, 0.1689726083361977, 3.4358531220869954, 1], [7, -40.52, 0.7945865381999999, 1.273274282321053, 0.2268597531696099, 3.256688497104738, 1], [8, -47.56, 0.9288488118, 0.0888858811515087, 1.6262093186242867, 8.970756386601588, 1], [8, -47.24, 0.8308176239999999, 0.09575334549396493, 1.418825627824654, 6.884462756465809, 2]]
