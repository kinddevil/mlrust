use rand::seq::SliceRandom;
use rand::thread_rng;
use rusty_machine::analysis::score::neg_mean_squared_error;
use rusty_machine::learning::SupModel;
use rusty_machine::learning::lin_reg::LinRegressor;
use rusty_machine::linalg::{Matrix, Vector};
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::path::Path;
use std::vec::Vec;

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>());
}

pub struct BostonHousing {
    crim: f64,
    zn: f64,
    indus: f64,
    chas: f64,
    nox: f64,
    rm: f64,
    age: f64,
    dis: f64,
    rad: f64,
    tax: f64,
    ptratio: f64,
    black: f64,
    lstat: f64,
    medv: f64,
}

impl BostonHousing {
    pub fn new(v: &[&str]) -> Option<BostonHousing> {
        if v.len() != 14 {
            return None;
        }

        let f64_formatted: Vec<f64> = v
            .iter()
            .map(|s| s.parse::<f64>())
            .collect::<Result<_, _>>()
            .ok()?;

        Some(BostonHousing {
            crim: f64_formatted[0],
            zn: f64_formatted[1],
            indus: f64_formatted[2],
            chas: f64_formatted[3],
            nox: f64_formatted[4],
            rm: f64_formatted[5],
            age: f64_formatted[6],
            dis: f64_formatted[7],
            rad: f64_formatted[8],
            tax: f64_formatted[9],
            ptratio: f64_formatted[10],
            black: f64_formatted[11],
            lstat: f64_formatted[12],
            medv: f64_formatted[13],
        })
    }
    pub fn into_feature_vector(&self) -> Vec<f64> {
        vec![
            self.crim,
            self.zn,
            self.indus,
            self.chas,
            self.nox,
            self.rm,
            self.age,
            self.dis,
            self.rad,
            self.tax,
            self.ptratio,
            self.black,
            self.lstat,
        ]
    }
    pub fn into_targets(&self) -> f64 {
        self.medv
    }
}

fn get_boston_record(s: &str) -> Option<BostonHousing> {
    let v: Vec<&str> = s.split(',').collect();
    BostonHousing::new(&v)
}

fn get_boston_records_from_file(fl: impl AsRef<Path>) -> Vec<BostonHousing> {
    let file = File::open(fl).expect("no such file");
    let buf = BufReader::new(file);
    buf.lines()
        .enumerate()
        .map(|(n, l)| l.expect(&format!("Could not parse line no {}", n)))
        .filter_map(|r| get_boston_record(&r))
        .collect()
}

fn r_squared_score(targets: &[f64], predictions: &[f64]) -> f64 {
    if targets.is_empty() || targets.len() != predictions.len() {
        return f64::NAN;
    }

    let target_mean = targets.iter().sum::<f64>() / targets.len() as f64;
    let residual_sum = targets
        .iter()
        .zip(predictions)
        .map(|(target, prediction)| (target - prediction).powi(2))
        .sum::<f64>();
    let total_sum = targets
        .iter()
        .map(|target| (target - target_mean).powi(2))
        .sum::<f64>();

    1.0 - residual_sum / total_sum
}

pub fn run() -> Result<(), Box<dyn Error>> {
    let fl = "data/housing.csv";
    let mut data = get_boston_records_from_file(&fl);
    println!("loaded {} complete records", data.len());

    data.shuffle(&mut thread_rng());
    let train_size = data.len() * 4 / 5;
    let test_size = data.len() - train_size;
    let (training_data, test_data) = data.split_at(train_size);

    let boston_x_train: Vec<f64> = training_data
        .iter()
        .flat_map(BostonHousing::into_feature_vector)
        .collect();
    let boston_y_train: Vec<f64> = training_data
        .iter()
        .map(BostonHousing::into_targets)
        .collect();
    let boston_x_test: Vec<f64> = test_data
        .iter()
        .flat_map(BostonHousing::into_feature_vector)
        .collect();
    let boston_y_test: Vec<f64> = test_data.iter().map(BostonHousing::into_targets).collect();

    let boston_x_train = Matrix::new(train_size, 13, boston_x_train);
    let boston_y_train = Vector::new(boston_y_train);
    let boston_x_test = Matrix::new(test_size, 13, boston_x_test);
    let boston_y_test = Matrix::new(test_size, 1, boston_y_test);

    let mut lin_model = LinRegressor::default();
    lin_model.train(&boston_x_train, &boston_y_train)?;
    let predictions = lin_model.predict(&boston_x_test)?;
    let predictions = Matrix::new(test_size, 1, predictions.into_vec());
    let acc = neg_mean_squared_error(&predictions, &boston_y_test);
    println!("linear regression error: {:?}", acc);
    println!(
        "linear regression R2 score: {:?}",
        r_squared_score(boston_y_test.data(), predictions.data())
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let x = "learning rust";
    println!("{}", x);
    print_type_of(&x);

    let x = "learning rust";
    let y = 6;
    let z = 3.14;
    println!("{}", x);
    println!("type of x:");
    print_type_of(&x);
    println!("type of y:");
    print_type_of(&y);
    println!("type of z:");
    print_type_of(&z);

    run()
}
