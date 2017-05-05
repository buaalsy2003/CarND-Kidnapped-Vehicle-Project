/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random> // Need this for sampling from distributions
#include <algorithm>
#include <iostream>
#include <numeric>
#include "particle_filter.h"

using namespace std;

default_random_engine gen;

//void AddRandomGaussianNoise(double &x, double &y, double &theta)
//{
//	default_random_engine gen;
//	gen.seed(101);
//
//	x = dist_x(gen);
//	y = dist_y(gen);
//	theta = dist_theta(gen);
//}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	gen.seed(101);
	num_particles = 100; 
	is_initialized = true;

	// TODO: Create normal distributions for x, y and theta.
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	particles.resize(num_particles); // Resize the `particles` vector to fit desired number of particles
	weights.resize(num_particles);
	double initWeight = 1.0 / num_particles;
	for (int i = 0; i < num_particles; i++) {
		particles[i].id = i;
		//Add noise
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
		particles[i].weight = initWeight;
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	normal_distribution<double> dist_x(0.0, std_pos[0]);
	normal_distribution<double> dist_y(0.0, std_pos[1]);
	normal_distribution<double> dist_theta(0.0, std_pos[2]);

	double yaw_d_t = yaw_rate * delta_t;
	double vel_yaw = velocity / yaw_rate;

	for (auto& p: particles) {
		const double theta_new = p.theta + yaw_d_t;
		p.x += vel_yaw * (sin(theta_new) - sin(p.theta));
		p.y += vel_yaw * (cos(p.theta) - cos(theta_new));
		p.theta = theta_new;

		//Add noise
		p.x += dist_x(gen);
		p.y += dist_y(gen);
		p.theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (auto& obs: observations) {
		double closest_dist = 1e9;
		int closest_id;
		for (auto& pred: predicted) {
			//Find the closest landmark and record its id
			double current_dist = dist(obs.x, obs.y, pred.x, pred.y);
			if (current_dist < closest_dist) {
				closest_dist = current_dist;
				closest_id = pred.id;
			}
		}
		//assigning closest landmark id to the observation
		obs.id = closest_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	double sum_w = 0.0; // Sum of weights for future weights normalization
	double tmp = 2 * M_PI * std_landmark[0] * std_landmark[1];
	for (auto& p: particles) {
		//transforming car observations to global map coordinates
		vector<LandmarkObs> car_observations;
		for (auto& obs: observations) {
			//reading current particle coordinates
			double xt = p.x, yt = p.y;
			double dThetaCos = cos(p.theta);
			double dThetaSin = sin(p.theta);
			//transforming observation to global coordinates
			xt += obs.x * dThetaCos - obs.y * dThetaSin;
			yt += obs.x * dThetaSin + obs.y * dThetaCos;
			LandmarkObs observation = { -1, xt, yt };
			car_observations.push_back(observation);
		}

		//creating local map of landmarks
		vector<LandmarkObs> landmarkInRange;
		int id = 0;
		for (auto& lm: map_landmarks.landmark_list) {
			//get landmarks in the range of the car sensor
			if (dist(lm.x_f, lm.y_f, p.x, p.y) <= sensor_range) {
				LandmarkObs landmark = { id++, lm.x_f, lm.y_f };
				landmarkInRange.push_back(landmark);
			}
		}

		//assigning to each observation id of the corresponding landmark
		dataAssociation(landmarkInRange, car_observations);

		double weight = 1.0;
		for (auto& obs: car_observations) {
			LandmarkObs land_obs = landmarkInRange[obs.id];
			double p_x = pow(obs.x - land_obs.x, 2) / (2 * pow(std_landmark[0], 2));
			double p_y = pow(obs.y - land_obs.y, 2) / (2 * pow(std_landmark[1], 2));
			double power = -1.0 * (p_x + p_y);
			weight *= exp(power);
		}
		p.weight = weight;
		sum_w += weight;
	}

	for (int i = 0; i < num_particles; i++) {
		particles[i].weight /= sum_w * tmp;
		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	discrete_distribution<> distParticles(weights.begin(), weights.end());
	vector<Particle> newParticles;
	for (int i = 0; i < num_particles; i++) {
		Particle newP = particles[distParticles(gen)];
		newParticles.push_back(newP);
	}
	particles.clear();
	particles = newParticles;

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}