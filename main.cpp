#include <iostream>
#include <algorithm>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <random>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

template <typename T>
constexpr const T& clamp(const T& value, const T& min, const T& max) {
    return (value < min) ? min : ((value > max) ? max : value);
}

void gaussian_filtering(float* in_image, 
                        int w, int h, 
                        float sigma_p, 
                        const char* out_filename = "guassian.hdr") {
    std::vector<float> out_image(w * h * 3, 0.0f);

    for (int x_i = 0; x_i < w; x_i++)
    for (int y_i = 0; y_i < h; y_i++) {
        float sum_of_weights = 0.0f;
        float sum_of_weighted_values[3] = {0.0f, 0.0f, 0.0f};

        for (int dx = -sigma_p; dx <= sigma_p; dx++)
        for (int dy = -sigma_p; dy <= sigma_p; dy++) {
            int x_j = std::max(0, std::min(w - 1, x_i + dx));
            int y_j = std::max(0, std::min(h - 1, y_i + dy));
            float weight = std::exp(-(dx * dx + dy * dy) / (2 * sigma_p * sigma_p));
            for (int channel = 0; channel < 3; channel++)
                sum_of_weighted_values[channel] += weight * in_image[(x_j + y_j * w) * 3 + channel];
            sum_of_weights += weight;
        }

        for (int channel = 0; channel < 3; channel++) {
            if (sum_of_weights == 0)
                out_image[(x_i + y_i * w) * 3 + channel] = 0.0f;
            out_image[(x_i + y_i * w) * 3 + channel] = static_cast<float>(sum_of_weighted_values[channel] / sum_of_weights);
        }
    }

    stbi_write_hdr(out_filename, w, h, 3, out_image.data());
}


void bilateral_filtering(float* in_image, 
                         int w, int h, 
                         float sigma_p, float sigma_c, 
                         const char* out_filename = "bilateral.hdr") {
    std::vector<float> out_image(w * h * 3, 0.0);

    for (int x_i = 0; x_i < w; x_i++)
    for (int y_i = 0; y_i < h; y_i++) {
        float sum_of_weights = 0.0f;
        float sum_of_weighted_values[3] = {0.0f, 0.0f, 0.0f};

        for (int dx = -sigma_p; dx <= sigma_p; dx++)
        for (int dy = -sigma_p; dy <= sigma_p; dy++) {
            int x_j = std::max(0, std::min(w - 1, x_i + dx));
            int y_j = std::max(0, std::min(h - 1, y_i + dy));

            float color_distance_2 = 0.0f;
            for (int channel = 0; channel < 3; channel++)
                color_distance_2 += std::pow(std::abs(in_image[(x_j + y_j * w) * 3 + channel] - in_image[(x_i + y_i * w) * 3 + channel]), 2);

            float weight = std::exp(-(dx * dx + dy * dy) / (2 * sigma_p * sigma_p) - color_distance_2 / (2 * sigma_c * sigma_c));
            for (int channel = 0; channel < 3; channel++)
                sum_of_weighted_values[channel] += weight * in_image[(x_j + y_j * w) * 3 + channel];
            sum_of_weights += weight;
        }
        
        for (int channel = 0; channel < 3; channel++) {
            if (sum_of_weights == 0)
                out_image[(x_i + y_i * w) * 3 + channel] = 0.0f;
            out_image[(x_i + y_i * w) * 3 + channel] = static_cast<float>(sum_of_weighted_values[channel] / sum_of_weights);
        }
    }
    
    stbi_write_hdr(out_filename, w, h, 3, out_image.data());
}

void joint_bilateral_filtering(float* origin, 
                               float* albedo, 
                               float* normal, 
                               int w, int h, 
                               float sigma_p, float sigma_c, float sigma_n, float sigma_a, 
                               const char* out_filename = "joint_bilateral.hdr") {
    std::vector<float> out_image(w * h * 3, 0.0f);

    for (int x_i = 0; x_i < w; x_i++)
    for (int y_i = 0; y_i < h; y_i++) {
        float sum_of_weights = 0.0f;
        float sum_of_weighted_values[3] = {0.0f, 0.0f, 0.0f};

        for (int dx = -sigma_p; dx <= sigma_p; dx++)
        for (int dy = -sigma_p; dy <= sigma_p; dy++) {
            int x_j = std::max(0, std::min(w - 1, x_i + dx));
            int y_j = std::max(0, std::min(h - 1, y_i + dy));

            float color_distance_2 = 0.0f;
            for (int channel = 0; channel < 3; channel++)
                color_distance_2 += std::pow(std::abs(origin[(x_j + y_j * w) * 3 + channel]  - origin[(x_i + y_i * w) * 3 + channel]), 2);

            float dot_product = 0.0f;
            for (int channel = 0; channel < 3; channel++) {
                float normal_j = 2.0f * (normal[(x_j + y_j * w) * 3 + channel]) - 1.0f;
                float normal_i = 2.0f * (normal[(x_i + y_i * w) * 3 + channel]) - 1.0f;
                dot_product += normal_j * normal_i;
            }
            dot_product = clamp(dot_product, 0.0f, 1.0f);
            float normal_distance_2 = std::pow(std::acos(dot_product) / M_PI, 2);

            float albedo_difference_2 = 0.0f;
            for (int channel = 0; channel < 3; channel++)
                albedo_difference_2 += std::pow(std::abs(albedo[(x_j + y_j * w) * 3 + channel] - albedo[(x_i + y_i * w) * 3 + channel]), 2);

            float weight = std::exp(-(dx * dx + dy * dy) / (2 * sigma_p * sigma_p) 
                - color_distance_2 / (2 * sigma_c * sigma_c)
                - normal_distance_2 / (2 * sigma_n * sigma_n)
                - albedo_difference_2 / (2 * sigma_a * sigma_a));
            for (int channel = 0; channel < 3; channel++)
                sum_of_weighted_values[channel] += weight * origin[(x_j + y_j * w) * 3 + channel];
            sum_of_weights += weight;
        }
        
        for (int channel = 0; channel < 3; channel++) {
            if (sum_of_weights == 0)
                out_image[(x_i + y_i * w) * 3 + channel] = 0.0f;
            out_image[(x_i + y_i * w) * 3 + channel] = static_cast<float>(sum_of_weighted_values[channel] / sum_of_weights);
        }
    }
    
    stbi_write_hdr(out_filename, w, h, 3, out_image.data());
}

// g++ -o main main.cpp --no-warnings --std=c++11
int main() { 
    const std::string subject = "kitchen";
    float sigma_p = 5.0f;
	float sigma_c = 0.7f;
	float sigma_n = 0.2f;
	float sigma_a = 0.2f;

    // const std::string subject = "spaceship";
	// float sigma_p = 3.0f;
	// float sigma_c = 0.5f;
	// float sigma_n = 0.1f;
	// float sigma_a = 0.1f;

    // const std::string subject = "veach-ajar";
    // float sigma_p = 6.0f;
	// float sigma_c = 0.5f;
	// float sigma_n = 0.2f;
	// float sigma_a = 0.1f;

    int w_origin, h_origin;
    float *origin = stbi_loadf(("images/" + subject + "/origin.hdr").c_str(), &w_origin, &h_origin, NULL, 3);   
    if (origin == nullptr) {
        std::cout << "Invalid origin image!" << std::endl;
        return 0;
    } 

    int w_albedo, h_albedo;
    float *albedo = stbi_loadf(("images/" + subject + "/albedo.hdr").c_str(), &w_albedo, &h_albedo, NULL, 3);   
    if (albedo == nullptr) {
        std::cout << "Invalid albedo image!" << std::endl;
        return 0;
    }

    int w_normal, h_normal;
    float *normal = stbi_loadf(("images/" + subject + "/normal.hdr").c_str(), &w_normal, &h_normal, NULL, 3);   
    if (normal == nullptr) {
        std::cout << "Invalid normal image!" << std::endl;
        return 0;
    }

    gaussian_filtering(origin, w_origin, h_origin, sigma_p);
    bilateral_filtering(origin, w_origin, h_origin, sigma_p, sigma_c);
    joint_bilateral_filtering(origin, albedo, normal, w_origin, h_origin, sigma_p, sigma_c, sigma_n, sigma_a);
    
    return 0;
}
