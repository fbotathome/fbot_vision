#ifndef LASER_PEOPLE_DETECTOR_PERSON_TRACKER_HPP
#define LASER_PEOPLE_DETECTOR_PERSON_TRACKER_HPP

#include <cstdint>
#include <vector>
#include <map>
#include <memory>
#include <cmath>
#include <algorithm>
#include <rclcpp/rclcpp.hpp>
#include <fbot_vision_msgs/msg/person_detection.hpp>

namespace laser_people_detector {

using PersonDetection = fbot_vision_msgs::msg::PersonDetection;

/**
 * @brief Represents a tracked person over time with confidence scoring and prediction
 */
struct TrackedPerson {
	int32_t track_id;               // Unique ID for this person
	double x, y;                    // Current position
	double vx, vy;                  // Estimated velocity (motion prediction)
	double confidence;              // Confidence score (0.0-1.0)
	uint32_t frame_count;           // Number of frames in this track
	uint32_t consecutive_misses;    // Number of frames since last detection
	uint32_t detections_count;      // Total number of detections for this track
	rclcpp::Time last_update;       // Timestamp of last update
	rclcpp::Time first_detection;   // Timestamp of first detection
	double velocity_confidence;     // Confidence in velocity estimate (affected by occlusion)

	TrackedPerson(int32_t id, double pos_x, double pos_y, rclcpp::Time time)
		: track_id(id), x(pos_x), y(pos_y), vx(0.0), vy(0.0),
		  confidence(0.3), frame_count(1), consecutive_misses(0),
		  detections_count(1), last_update(time), first_detection(time),
		  velocity_confidence(0.0) {}

	double distance_to(double px, double py) const {
		double dx = x - px;
		double dy = y - py;
		return std::sqrt(dx * dx + dy * dy);
	}

	/**
	 * @brief Predict position based on velocity estimate
	 */
	void predict_position(double time_delta) {
		if (time_delta > 0.0 && velocity_confidence > 0.2) {
			x += vx * time_delta;
			y += vy * time_delta;
		}
	}

	/**
	 * @brief Update track with new detection and recalculate confidence
	 */
	void update(double pos_x, double pos_y, rclcpp::Time time) {
		double dt = (time - last_update).seconds();
		
		// Update velocity estimate (low-pass filtered)
		if (dt > 0.0 && dt < 1.0) {  // Ignore unrealistic time deltas
			double new_vx = (pos_x - x) / dt;
			double new_vy = (pos_y - y) / dt;
			
			// Only update velocity if it seems reasonable (filter outliers)
			double speed = std::sqrt(new_vx * new_vx + new_vy * new_vy);
			if (speed < 3.0) {  // Max reasonable speed: 3 m/s
				// Exponential moving average for velocity (alpha = 0.5)
				if (detections_count > 1) {
					vx = 0.5 * vx + 0.5 * new_vx;
					vy = 0.5 * vy + 0.5 * new_vy;
				} else {
					vx = new_vx;
					vy = new_vy;
				}
			}
		}

		x = pos_x;
		y = pos_y;
		frame_count++;
		consecutive_misses = 0;
		detections_count++;
		velocity_confidence = std::min(1.0, velocity_confidence + 0.15);
		last_update = time;
		
		// Update confidence (increases with detections, capped at track age penalty)
		update_confidence(true);
	}

	/**
	 * @brief Mark a missed detection and adjust confidence
	 */
	void mark_miss() {
		consecutive_misses++;
		velocity_confidence *= 0.8;  // Reduce confidence in velocity during occlusion
		frame_count++;
		update_confidence(false);
	}

	/**
	 * @brief Update confidence score based on track history
	 */
	void update_confidence(bool was_detected) {
		// Base confidence: increases with detection ratio and age
		double detection_ratio = static_cast<double>(detections_count) / frame_count;
		double age_boost = std::min(1.0, frame_count / 30.0);  // Max boost at 30 frames
		double base_confidence = 0.3 + (detection_ratio * 0.5) + (age_boost * 0.2);
		
		// Occlusion penalty: reduce confidence during consecutive misses
		double occlusion_penalty = std::pow(0.85, consecutive_misses);
		
		confidence = base_confidence * occlusion_penalty;
		confidence = std::max(0.0, std::min(1.0, confidence));
	}

	/**
	 * @brief Get association score combining distance and prediction
	 * @param detection_x Detection X position
	 * @param detection_y Detection Y position
	 * @param time Current timestamp
	 * @return Score (lower is better), -1 if too far
	 */
	double get_association_score(double detection_x, double detection_y, 
	                             double max_distance, double prediction_weight = 0.3) {
		double current_dist = distance_to(detection_x, detection_y);
		
		if (current_dist > max_distance) {
			return -1.0;  // Too far, can't associate
		}
		
		// Combine actual distance with prediction-based penalty
		// Prediction helps maintain association even if detection moves unexpectedly
		double score = current_dist * (1.0 - prediction_weight * velocity_confidence);
		return score;
	}
};

/**
 * @brief Manages tracking of multiple people across frames with robust data association
 */
class PersonTracker {
private:
	std::map<int32_t, TrackedPerson> m_tracks;
	int32_t m_next_track_id{1};
	double m_max_tracking_distance{1.5};      // Max distance to associate detection to track
	double m_occlusion_distance{2.5};         // Extended distance during occlusion
	uint32_t m_track_timeout_frames{15};      // Frames before removing unmatched track
	uint32_t m_occlusion_timeout_frames{25};  // Extended timeout for temporarily occluded tracks
	uint32_t m_min_track_age{2};              // Minimum frames before considering track valid
	double m_confidence_threshold{0.3};       // Minimum confidence to consider track
	rclcpp::Logger m_logger;

public:
	PersonTracker(rclcpp::Logger logger = rclcpp::get_logger("PersonTracker"))
		: m_logger(logger) {}

	void set_max_tracking_distance(double distance) {
		m_max_tracking_distance = distance;
		m_occlusion_distance = distance * 1.8;
	}

	void set_track_timeout_frames(uint32_t frames) {
		m_track_timeout_frames = frames;
		m_occlusion_timeout_frames = frames * 2;  // Extended for occluded tracks
	}

	void set_min_track_age(uint32_t frames) {
		m_min_track_age = frames;
	}

	void set_confidence_threshold(double threshold) {
		m_confidence_threshold = threshold;
	}

	/**
	 * @brief Predict next positions for all tracks
	 */
	void predict_all(double time_delta) {
		for (auto& track : m_tracks) {
			track.second.predict_position(time_delta);
		}
	}

	/**
	 * @brief Update tracks with new detections using robust multi-stage data association
	 * @param detections Current detections from the sensor
	 * @param timestamp Current timestamp
	 * @return Vector of updated tracked persons
	 */
	std::vector<TrackedPerson> update(
		const std::vector<PersonDetection>& detections,
		rclcpp::Time timestamp)
	{
		// Stage 1: Predict positions based on velocity
		if (!m_tracks.empty()) {
			static rclcpp::Time last_predict_time = timestamp;
			double dt = (timestamp - last_predict_time).seconds();
			if (dt > 0.0 && dt < 0.5) {
				predict_all(dt);
			}
			last_predict_time = timestamp;
		}

		std::vector<bool> matched_detections(detections.size(), false);
		std::vector<int32_t> matched_tracks;
		std::vector<std::pair<int32_t, size_t>> associations;  // track_id -> detection_idx

		// Stage 2: Multi-stage data association
		// First pass: Match high-confidence tracks with nearby detections
		for (auto it = m_tracks.begin(); it != m_tracks.end(); ++it) {
			if (it->second.confidence < 0.4) continue;  // Skip low-confidence tracks in first pass

			double min_score = m_max_tracking_distance;
			int best_detection_idx = -1;

			for (size_t d_idx = 0; d_idx < detections.size(); d_idx++) {
				if (matched_detections[d_idx]) continue;

				const auto& detection = detections[d_idx];
				double score = it->second.get_association_score(
					detection.position.x, detection.position.y, m_max_tracking_distance, 0.3);

				if (score >= 0.0 && score < min_score) {
					min_score = score;
					best_detection_idx = d_idx;
				}
			}

			if (best_detection_idx != -1) {
				matched_detections[best_detection_idx] = true;
				matched_tracks.push_back(it->first);
				associations.push_back({it->first, best_detection_idx});
			}
		}

		// Second pass: Match remaining low-confidence or occluded tracks with expanded distance
		for (auto it = m_tracks.begin(); it != m_tracks.end(); ++it) {
			if (std::find(matched_tracks.begin(), matched_tracks.end(), it->first) != matched_tracks.end()) {
				continue;  // Already matched
			}
			if (it->second.confidence >= 0.4) continue;  // High-confidence tracks already handled

			double min_score = m_occlusion_distance;
			int best_detection_idx = -1;

			for (size_t d_idx = 0; d_idx < detections.size(); d_idx++) {
				if (matched_detections[d_idx]) continue;

				const auto& detection = detections[d_idx];
				double score = it->second.get_association_score(
					detection.position.x, detection.position.y, m_occlusion_distance, 0.2);

				if (score >= 0.0 && score < min_score) {
					min_score = score;
					best_detection_idx = d_idx;
				}
			}

			if (best_detection_idx != -1) {
				matched_detections[best_detection_idx] = true;
				matched_tracks.push_back(it->first);
				associations.push_back({it->first, best_detection_idx});
			}
		}

		// Stage 3: Update matched tracks
		for (const auto& assoc : associations) {
			auto it = m_tracks.find(assoc.first);
			if (it != m_tracks.end()) {
				const auto& detection = detections[assoc.second];
				it->second.update(detection.position.x, detection.position.y, timestamp);
			}
		}

		// Stage 4: Handle unmatched tracks (increase miss counter, may predict position)
		std::vector<int32_t> tracks_to_remove;
		for (auto it = m_tracks.begin(); it != m_tracks.end(); ++it) {
			if (std::find(matched_tracks.begin(), matched_tracks.end(), it->first) == matched_tracks.end()) {
				it->second.mark_miss();
				
				// Remove track only if timeout exceeded
				uint32_t timeout = (it->second.confidence > 0.5) ? 
					m_occlusion_timeout_frames : m_track_timeout_frames;
				
				if (it->second.consecutive_misses > timeout) {
					tracks_to_remove.push_back(it->first);
					RCLCPP_DEBUG(m_logger, "Removing track %d (timeout after %u misses, confidence: %.2f)",
						it->first, it->second.consecutive_misses, it->second.confidence);
				}
			}
		}

		// Remove old tracks
		for (int32_t track_id : tracks_to_remove) {
			m_tracks.erase(track_id);
		}

		// Stage 5: Create new tracks for unmatched detections
		for (size_t d_idx = 0; d_idx < detections.size(); d_idx++) {
			if (!matched_detections[d_idx]) {
				const auto& detection = detections[d_idx];
				m_tracks.emplace(m_next_track_id, 
					TrackedPerson(m_next_track_id, detection.position.x, detection.position.y, timestamp));
				RCLCPP_DEBUG(m_logger, "Creating new track %d at (%.2f, %.2f)",
					m_next_track_id, detection.position.x, detection.position.y);
				m_next_track_id++;
			}
		}

		// Return valid tracks (with minimum age and confidence)
		std::vector<TrackedPerson> valid_tracks;
		for (const auto& track : m_tracks) {
			if (track.second.frame_count >= m_min_track_age && 
			    track.second.confidence >= m_confidence_threshold) {
				valid_tracks.push_back(track.second);
			}
		}

		return valid_tracks;
	}

	/**
	 * @brief Get a specific track by ID (returns only if confidence sufficient)
	 * @param track_id ID of the track to retrieve
	 * @return Pointer to tracked person or nullptr if not found or low confidence
	 */
	TrackedPerson* get_track(int32_t track_id) {
		auto it = m_tracks.find(track_id);
		if (it != m_tracks.end() && it->second.confidence >= m_confidence_threshold) {
			return &it->second;
		}
		return nullptr;
	}

	/**
	 * @brief Get all current tracks regardless of confidence
	 */
	std::vector<TrackedPerson> get_all_tracks() const {
		std::vector<TrackedPerson> result;
		for (const auto& track : m_tracks) {
			result.push_back(track.second);
		}
		return result;
	}

	/**
	 * @brief Get only valid (mature and confident) tracks
	 */
	std::vector<TrackedPerson> get_valid_tracks() const {
		std::vector<TrackedPerson> result;
		for (const auto& track : m_tracks) {
			if (track.second.frame_count >= m_min_track_age && 
			    track.second.confidence >= m_confidence_threshold) {
				result.push_back(track.second);
			}
		}
		return result;
	}

	/**
	 * @brief Get all high-confidence tracks (good for following)
	 */
	std::vector<TrackedPerson> get_high_confidence_tracks(double min_confidence = 0.6) const {
		std::vector<TrackedPerson> result;
		for (const auto& track : m_tracks) {
			if (track.second.frame_count >= m_min_track_age && 
			    track.second.confidence >= min_confidence) {
				result.push_back(track.second);
			}
		}
		return result;
	}

	/**
	 * @brief Clear all tracks
	 */
	void reset() {
		m_tracks.clear();
		m_next_track_id = 1;
	}

	/**
	 * @brief Get number of active tracks
	 */
	size_t get_track_count() const {
		return m_tracks.size();
	}

	/**
	 * @brief Get number of valid (confident) tracks
	 */
	size_t get_valid_track_count() const {
		size_t count = 0;
		for (const auto& track : m_tracks) {
			if (track.second.frame_count >= m_min_track_age && 
			    track.second.confidence >= m_confidence_threshold) {
				count++;
			}
		}
		return count;
	}
};

}  // namespace laser_people_detector

#endif  // LASER_PEOPLE_DETECTOR_PERSON_TRACKER_HPP
