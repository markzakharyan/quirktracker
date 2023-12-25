use crate::knappen::{run_point, ATLAS_RADII_PIXEL};
use csv::WriterBuilder;
use nalgebra::Vector4;
use serde::Deserialize;
use std::env;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Deserialize)]
struct EventState {
    PID: f32,
    E: f64,
    px: f64,
    py: f64,
    pz: f64,
    m: f64,
    q: f32,
    status: f32,
    #[serde(rename = "Event#")]
    event_id: f32,
}

pub fn process(lambda: f64, passed: &Vec<i32>, input_path: &str, output_path: &str) {
    let mut input = csv::ReaderBuilder::new().has_headers(true).from_path(input_path).unwrap();

    let event_states: Vec<EventState> = input.deserialize().map(|result| result.unwrap()).collect();

    let mut data: Vec<Vec<String>> = vec![
        vec!["EventID", "TruthID", "PID", "Layer", "r[cm]", "phi", "z[cm]", "E[GeV]", "px[GeV]", "py[GeV]", "pz[GeV]"]
            .iter().map(|&s| s.to_string()).collect(),
    ];

    for (ii, event_state) in event_states.iter().enumerate().take(88860) {

        if !passed.contains(unsafe { &event_state.event_id.to_int_unchecked() }) {
            continue;
        }

        let vec1 = Vector4::new(event_state.E, event_state.px, event_state.py, event_state.pz);
        let vec2 = vec1.clone();

        let truth_id = ii + 2;

        println!("truth id: {}", truth_id);


        // TODO: Investigate why program hangs at truth_id=383
        let aa: Vec<(i32, f64, f64, f64, f64, f64, i32)> = run_point(&vec1, &vec2, lambda, false, false);

        if aa.len() == 16 {
            for i in 1..9 {
                let select: usize;
                if event_state.PID > 0.0 {
                    select = 2 * (i - 1);
                } else {
                    select = 2 * i - 1;
                }

                let layer = aa[select].0;
                let r = ATLAS_RADII_PIXEL[(layer - 1) as usize];
                let z = aa[select].1;
                let phi = aa[select].2;

                let record = vec![
                    // listhere = [EventID, TruthID, PID, layer, r, phi, z] + list(vecs[0])
                    (event_state.event_id).to_string(),
                    truth_id.to_string(),
                    event_state.PID.to_string(),
                    layer.to_string(),
                    r.to_string(),
                    phi.to_string(),
                    z.to_string(),
                    vec1[0].to_string(),
                    vec1[1].to_string(),
                    vec1[2].to_string(),
                    vec1[3].to_string(),
                ];

                if phi.is_finite() && z.is_finite() {
                    data.push(record);
                }
            }
            
        }


    }

    std::fs::create_dir_all(Path::new(output_path).parent().unwrap()).unwrap();
    let mut wtr = WriterBuilder::new().from_path(output_path).unwrap();
    for row in data {
        wtr.write_record(&row).unwrap();
    }
    wtr.flush().unwrap();

}

pub fn _default_background() {
    let lambda = 20.0;

    let current_directory = env::current_dir().unwrap();
    
    let input_path = format!("{}/4vector_pionbgd_wCuts.csv", current_directory.display());
    let output_path = format!("{}/HitFiles/Bgd/Bgd_500_1jet_wCuts.csv", current_directory.display());

    // passed list provided here only for debugging purposes
    let passed: Vec<i32> = vec![3, 4, 10, 14, 15, 19, 20, 21, 26, 31, 32, 33, 40, 41, 43, 47, 50, 51, 52, 55, 56, 57, 58, 59, 61, 66, 67, 68, 69, 70, 79, 81, 82, 83, 87, 88, 94, 95, 98, 100, 103, 104, 105, 111, 113, 114, 116, 117, 118, 119, 127, 128, 131, 134, 136, 139, 140, 141, 142, 143, 144, 150, 152, 154, 155, 165, 166, 171, 173, 174, 176, 177, 178, 180, 181, 184, 186, 187, 188, 189, 192, 194, 195, 197, 198, 199, 203, 205, 208, 211, 212, 220, 222, 224, 228, 229, 234, 236, 237, 240];


    let start_time = Instant::now();
    let passed = process(lambda, &passed, &input_path, &output_path);
    let duration = start_time.elapsed();

    println!("(Rust Background Regular) Time taken: {:?} seconds", duration.as_secs_f32());
    println!("passed: {:?}", passed);
}

// (Quirk Regular) Time taken: 53.58206 seconds
// passed: [4, 5, 7, 8, 9, 10, 11, 12, 18, 20, 21, 25, 26, 27, 32, 33, 42, 44, 46, 47, 49, 50, 51, 52, 54, 56, 57, 58, 60, 70, 73, 75, 77, 79, 81, 82, 84, 85, 87, 89, 92, 95, 96, 102, 106, 107, 108, 109, 110, 112, 115, 122, 123, 126, 127, 128, 129, 131, 132, 139, 140, 147, 155, 158, 159, 160, 164, 170, 172, 173, 174, 175, 180, 182, 183, 184, 185, 188, 190, 191, 195, 197, 199, 201, 205, 206, 211, 215, 218, 220, 222, 223, 225, 226, 227, 231, 234, 235, 240, 241, 246, 248, 249, 254, 255, 257, 260, 261, 262, 266, 268, 275, 277, 278, 281, 282, 283, 284, 285, 288, 291, 294, 296, 300, 301, 306, 307, 308, 309, 310, 317, 319, 321, 333, 340, 344, 347, 350, 352, 353, 359, 360, 363, 365, 370, 371, 373, 375, 378, 380, 382, 384, 387, 388, 392, 399, 401, 403, 407, 409, 416, 423, 424, 429, 436, 437, 441, 444, 446, 449, 451, 457, 458, 459, 461, 462, 465, 467, 468, 471, 476, 481, 485, 491, 493, 497, 499, 500]
