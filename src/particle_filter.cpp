/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define EPS 0.00001

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
    //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    
    num_particles = 100;

    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];

    default_random_engine gen;
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);	
    normal_distribution<double> dist_theta(theta, std_theta);

    for (int i = 0; i < num_particles; ++i) {
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        particles.push_back(std::move(p));
        weights.push_back(1.0);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    double std_x = std_pos[0];
    double std_y = std_pos[1];
    double std_theta = std_pos[2];

    default_random_engine gen;
    normal_distribution<double> x_noise(0, std_x);
    normal_distribution<double> y_noise(0, std_y);	
    normal_distribution<double> theta_noise(0, std_theta);

    for (auto & p : particles) {
        double x, y, theta;

        if (fabs(yaw_rate) < EPS) {
            theta = p.theta;
            x = p.x + velocity * delta_t * cos(theta);
            y = p.y + velocity * delta_t * sin(theta);
        } else {
            theta = p.theta + yaw_rate*delta_t;
            x = p.x + velocity/yaw_rate * (sin(theta) - sin(p.theta));
            y = p.y + velocity/yaw_rate * (cos(p.theta) - cos(theta));
        }

        p.x = x + x_noise(gen);
        p.y = y + y_noise(gen);
        p.theta = theta + theta_noise(gen);
        p.weight = 1.0;
    }
}

void ParticleFilter::dataAssociation(const std::vector<LandmarkObs>& predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.

    for (auto & obs : observations) {
        double min_dist = std::numeric_limits<double>::max();

        for (auto const& pred_obs : predicted) {
            double d = dist(obs.x, obs.y, pred_obs.x, pred_obs.y);
            if (d < min_dist) {
                min_dist = d;
                obs.id = pred_obs.id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
        const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation 
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    for (auto & p : particles) {
        // find map landmarks that are within sensor range as predicted landmarks
        std::vector<LandmarkObs> predicted;
        for (auto const& landmark : map_landmarks.landmark_list) {
            if (dist(p.x, p.y, landmark.x_f, landmark.y_f) <= sensor_range) {
                predicted.push_back(
                    LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
            }
        }

        // if all landmarks are outside of sensor range, 
        // this particular particle contains zero information 
        if (0 == predicted.size()) {
            p.weight = 0.0;
            weights.push_back(0.0);
            continue;
        }

        // transform observation into MAP's coordinate sstem
        std::vector<LandmarkObs> observation_map;

        for (auto const& obs : observations) {
            LandmarkObs obs_map;
            obs_map.x = p.x + cos(p.theta) * obs.x - sin(p.theta) * obs.y;
            obs_map.y = p.y + sin(p.theta) * obs.x + cos(p.theta) * obs.y;

            observation_map.push_back(obs_map);
        }

        dataAssociation(predicted, observation_map);

        vector<int> associations;
        vector<double> sense_x;
        vector<double> sense_y;

        // calculate weight
        double weight = 1.0;
        for (auto const& obs_map : observation_map) {
            double sig_x = std_landmark[0];
            double sig_y = std_landmark[1];

            // calculate normalization term
            double norm_term = 1/(2 * M_PI * sig_x * sig_y);

            // calculate exponent
            Map::single_landmark_s landmark = map_landmarks.landmark_list[obs_map.id-1];
            double x_term = pow(obs_map.x - landmark.x_f, 2) / (2 * pow(sig_x, 2));
            double y_term = pow(obs_map.y - landmark.y_f, 2) / (2 * pow(sig_y, 2));
            
            // calculate weight using normalization terms and exponent
            weight *= norm_term * exp(-(x_term + y_term));

            associations.push_back(landmark.id_i);
            sense_x.push_back(obs_map.x);
            sense_y.push_back(obs_map.y);
        }
        
        p.weight = weight;
        weights.push_back(weight);

        SetAssociations(p, associations, sense_x, sense_y);
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution


    std::discrete_distribution<double> d(weights.begin(), weights.end());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<Particle> p2;

    for (int i = 0; i < num_particles; ++i) {
        Particle p = particles[d(gen)];
        p2.push_back(p);
    }

    // applied move assignment operator
    particles = std::move(p2);
    weights.clear();
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
