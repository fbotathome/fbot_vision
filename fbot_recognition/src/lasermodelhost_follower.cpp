#include <math.h>
#include <algorithm>
#include <vector>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/int32.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <fbot_vision_msgs/msg/person_detection_list.hpp>

#include "person_tracker.hpp"

namespace laser_people_detector {

using geometry_msgs::msg::Twist;
using std_msgs::msg::Int32;
using visualization_msgs::msg::MarkerArray;
using fbot_vision_msgs::msg::PersonDetectionList;
using fbot_vision_msgs::msg::PersonDetection;

class LaserModelHostFollower : public rclcpp::Node {

	rclcpp::Subscription<PersonDetectionList>::SharedPtr m_subDetections{};
	rclcpp::Subscription<Int32>::SharedPtr m_subPersonSelection{};
	std::shared_ptr<rclcpp::Publisher<Twist>> m_pubVelocity{};
	std::shared_ptr<rclcpp::Publisher<MarkerArray>> m_pubTrackingMarkers{};

	// Tracking
	PersonTracker m_tracker;
	int32_t m_selected_track_id{-1};  // -1 means auto-select closest

	// Parameters
	double m_maxLinearVelocity{};
	double m_maxAngularVelocity{};
	double m_stoppingDistance{};
	double m_kpLinear{};   // Proportional gain for linear velocity
	double m_kdAngular{}; // Derivative/proportional gain for angular velocity
	double m_maxTrackingDistance{};
	int m_trackTimeoutFrames{};
	double m_confidenceThreshold{};
	bool m_publishTrackingMarkers{};
	bool m_preferHighConfidence{};  // Only follow high-confidence tracks during crowds

	// State
	double m_lastAngleError{0.0};
	std::string m_frame_id{"laser"};

public:

	LaserModelHostFollower(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
		: rclcpp::Node("laser_people_follower", options), 
		  m_tracker(this->get_logger())
	{
		// Declare parameters
		this->declare_parameter<double>("max_linear_velocity", 0.5);
		this->declare_parameter<double>("max_angular_velocity", 1.0);
		this->declare_parameter<double>("stopping_distance", 0.5);
		this->declare_parameter<double>("kp_linear", 1.0);
		this->declare_parameter<double>("kd_angular", 0.5);
		this->declare_parameter<double>("max_tracking_distance", 1.5);
		this->declare_parameter<int>("track_timeout_frames", 15);
		this->declare_parameter<double>("confidence_threshold", 0.3);
		this->declare_parameter<bool>("prefer_high_confidence", true);
		this->declare_parameter<bool>("publish_tracking_markers", false);
		this->declare_parameter<std::string>("detections_topic", "detected_people");
		this->declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
		this->declare_parameter<std::string>("person_selection_topic", "select_person");
		this->declare_parameter<std::string>("tracking_markers_topic", "tracked_people_markers");

		// Get parameters
		m_maxLinearVelocity = this->get_parameter("max_linear_velocity").as_double();
		m_maxAngularVelocity = this->get_parameter("max_angular_velocity").as_double();
		m_stoppingDistance = this->get_parameter("stopping_distance").as_double();
		m_kpLinear = this->get_parameter("kp_linear").as_double();
		m_kdAngular = this->get_parameter("kd_angular").as_double();
		m_maxTrackingDistance = this->get_parameter("max_tracking_distance").as_double();
		m_trackTimeoutFrames = this->get_parameter("track_timeout_frames").as_int();
		m_confidenceThreshold = this->get_parameter("confidence_threshold").as_double();
		m_preferHighConfidence = this->get_parameter("prefer_high_confidence").as_bool();
		m_publishTrackingMarkers = this->get_parameter("publish_tracking_markers").as_bool();

		std::string detectionsTopicName = this->get_parameter("detections_topic").as_string();
		std::string cmdVelTopicName = this->get_parameter("cmd_vel_topic").as_string();
		std::string personSelectionTopicName = this->get_parameter("person_selection_topic").as_string();
		std::string trackingMarkersTopicName = this->get_parameter("tracking_markers_topic").as_string();

		// Configure tracker
		m_tracker.set_max_tracking_distance(m_maxTrackingDistance);
		m_tracker.set_track_timeout_frames(static_cast<uint32_t>(m_trackTimeoutFrames));
		m_tracker.set_confidence_threshold(m_confidenceThreshold);

		// Create subscriptions and publishers
		m_subDetections = this->create_subscription<PersonDetectionList>(
			detectionsTopicName, 10,
			std::bind(&LaserModelHostFollower::onDetections, this, std::placeholders::_1));

		m_subPersonSelection = this->create_subscription<Int32>(
			personSelectionTopicName, 10,
			std::bind(&LaserModelHostFollower::onPersonSelection, this, std::placeholders::_1));

		m_pubVelocity = this->create_publisher<Twist>(cmdVelTopicName, 10);
		
		if (m_publishTrackingMarkers) {
			m_pubTrackingMarkers = this->create_publisher<MarkerArray>(trackingMarkersTopicName, 10);
		}

		RCLCPP_INFO(this->get_logger(), "[Laser People Follower] Initialized");
		RCLCPP_INFO(this->get_logger(), "  Max linear velocity: %.2f m/s", m_maxLinearVelocity);
		RCLCPP_INFO(this->get_logger(), "  Max angular velocity: %.2f rad/s", m_maxAngularVelocity);
		RCLCPP_INFO(this->get_logger(), "  Stopping distance: %.2f m", m_stoppingDistance);
		RCLCPP_INFO(this->get_logger(), "  Max tracking distance: %.2f m", m_maxTrackingDistance);
		RCLCPP_INFO(this->get_logger(), "  Track timeout: %d frames", m_trackTimeoutFrames);
		RCLCPP_INFO(this->get_logger(), "  Confidence threshold: %.2f", m_confidenceThreshold);
		RCLCPP_INFO(this->get_logger(), "  Prefer high confidence: %s", 
			m_preferHighConfidence ? "true" : "false");
		RCLCPP_INFO(this->get_logger(), "  Publishing tracking markers: %s", 
			m_publishTrackingMarkers ? "true" : "false");
	}

	void onDetections(const PersonDetectionList::SharedPtr msg)
	{
		if (msg->detections.empty()) {
			// No people detected, stop the robot
			publishVelocity(0.0, 0.0);
			return;
		}

		// Update tracker with new detections
		auto tracked_people = m_tracker.update(msg->detections, this->now());

		if (tracked_people.empty()) {
			// No valid tracks yet, stop the robot
			publishVelocity(0.0, 0.0);
			if (m_publishTrackingMarkers) {
				publishTrackingMarkers({}, msg->header.frame_id);
			}
			return;
		}

		// Get high-confidence tracks if in crowded environment
		std::vector<TrackedPerson> candidate_tracks;
		if (m_preferHighConfidence && m_tracker.get_valid_track_count() > 1) {
			// Multiple people detected, prefer high-confidence tracks
			candidate_tracks = m_tracker.get_high_confidence_tracks(0.5);
			if (candidate_tracks.empty()) {
				// Fall back to valid tracks if no high-confidence ones available
				candidate_tracks = tracked_people;
			}
		} else {
			// Single person or low-confidence preference mode
			candidate_tracks = tracked_people;
		}

		// Select target person
		TrackedPerson target_person = candidate_tracks[0];

		if (m_selected_track_id != -1) {
			// Try to follow the selected person
			auto selected = m_tracker.get_track(m_selected_track_id);
			if (selected != nullptr) {
				target_person = *selected;
				RCLCPP_DEBUG(this->get_logger(), 
					"[Laser People Follower] Following selected track %d (confidence: %.2f)", 
					m_selected_track_id, selected->confidence);
			} else {
				// Selected person not found, auto-select closest
				RCLCPP_WARN(this->get_logger(), 
					"[Laser People Follower] Selected track %d lost (possibly occluded), auto-selecting closest",
					m_selected_track_id);
				m_selected_track_id = -1;
				// Find closest from candidates
				double minDistance = target_person.distance_to(0.0, 0.0);
				for (const auto& person : candidate_tracks) {
					double distance = person.distance_to(0.0, 0.0);
					if (distance < minDistance) {
						minDistance = distance;
						target_person = person;
					}
				}
			}
		} else {
			// Auto-select closest person (from high-confidence candidates in crowds)
			double minDistance = target_person.distance_to(0.0, 0.0);
			for (const auto& person : candidate_tracks) {
				double distance = person.distance_to(0.0, 0.0);
				if (distance < minDistance) {
					minDistance = distance;
					target_person = person;
				}
			}
		}

		// Calculate linear velocity (based on distance)
		double distance_to_target = target_person.distance_to(0.0, 0.0);
		double distanceError = distance_to_target - m_stoppingDistance;
		double linearVelocity = m_kpLinear * distanceError;

		// Clamp to max velocity
		if (linearVelocity > m_maxLinearVelocity) {
			linearVelocity = m_maxLinearVelocity;
		} else if (linearVelocity < -m_maxLinearVelocity) {
			linearVelocity = -m_maxLinearVelocity;
		}

		// Calculate angular velocity (based on angle to target)
		double angleToTarget = std::atan2(target_person.y, target_person.x);
		double angleError = angleToTarget;
		
		double angularVelocity = m_kdAngular * angleError;

		// Clamp to max angular velocity
		if (angularVelocity > m_maxAngularVelocity) {
			angularVelocity = m_maxAngularVelocity;
		} else if (angularVelocity < -m_maxAngularVelocity) {
			angularVelocity = -m_maxAngularVelocity;
		}

		// If we're very close, stop
		if (distance_to_target < 0.1) {
			linearVelocity = 0.0;
		}

		publishVelocity(linearVelocity, angularVelocity);

		if (m_publishTrackingMarkers) {
			publishTrackingMarkers(tracked_people, msg->header.frame_id);
		}

		RCLCPP_DEBUG(this->get_logger(),
			"[Laser People Follower] Following track %d at (%.2f, %.2f) m, distance=%.2f m. Velocity: lin=%.2f, ang=%.2f",
			target_person.track_id, target_person.x, target_person.y, distance_to_target, 
			linearVelocity, angularVelocity);
	}

	void onPersonSelection(const Int32::SharedPtr msg)
	{
		if (msg->data == -1) {
			RCLCPP_INFO(this->get_logger(), 
				"[Laser People Follower] Auto-selecting closest person");
			m_selected_track_id = -1;
		} else {
			RCLCPP_INFO(this->get_logger(), 
				"[Laser People Follower] User selected track %d", msg->data);
			m_selected_track_id = msg->data;
		}
	}

	void publishVelocity(double linearVelocity, double angularVelocity)
	{
		auto msg = std::make_unique<Twist>();
		msg->linear.x = linearVelocity;
		msg->linear.y = 0.0;
		msg->linear.z = 0.0;
		msg->angular.x = 0.0;
		msg->angular.y = 0.0;
		msg->angular.z = angularVelocity;
		m_pubVelocity->publish(std::move(msg));
	}

	void publishTrackingMarkers(const std::vector<TrackedPerson>& tracked_people, 
	                             const std::string& frame_id)
	{
		if (!m_pubTrackingMarkers) {
			return;
		}

		auto marker_array = std::make_unique<MarkerArray>();

		// Clear previous markers
		visualization_msgs::msg::Marker delete_marker;
		delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
		marker_array->markers.push_back(delete_marker);

		// Create markers for tracked people
		for (const auto& person : tracked_people) {
			visualization_msgs::msg::Marker marker;
			marker.header.frame_id = frame_id;
			marker.header.stamp = this->now();
			marker.id = person.track_id;
			marker.type = visualization_msgs::msg::Marker::CYLINDER;
			marker.action = visualization_msgs::msg::Marker::ADD;
			marker.pose.position.x = person.x;
			marker.pose.position.y = person.y;
			marker.pose.position.z = 0.5;
			marker.scale.x = 0.3;  // Cylinder diameter
			marker.scale.y = 0.3;
			marker.scale.z = 1.0;  // Cylinder height

			// Color based on confidence (red=low, yellow=medium, green=high)
			if (person.confidence < 0.4) {
				marker.color.r = 1.0;  // Red
				marker.color.g = 0.0;
				marker.color.b = 0.0;
			} else if (person.confidence < 0.6) {
				marker.color.r = 1.0;  // Yellow
				marker.color.g = 1.0;
				marker.color.b = 0.0;
			} else {
				marker.color.r = 0.0;  // Green
				marker.color.g = 1.0;
				marker.color.b = 0.0;
			}
			marker.color.a = 0.7;

			marker_array->markers.push_back(marker);

			// Add text label with track ID and confidence
			visualization_msgs::msg::Marker text_marker;
			text_marker.header.frame_id = frame_id;
			text_marker.header.stamp = this->now();
			text_marker.id = person.track_id + 10000;  // Offset to avoid ID conflicts
			text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
			text_marker.action = visualization_msgs::msg::Marker::ADD;
			text_marker.pose.position.x = person.x;
			text_marker.pose.position.y = person.y;
			text_marker.pose.position.z = 1.2;
			text_marker.scale.z = 0.2;
			text_marker.color.r = 1.0;
			text_marker.color.g = 1.0;
			text_marker.color.b = 1.0;
			text_marker.color.a = 1.0;
			text_marker.text = "ID:" + std::to_string(person.track_id) + 
			                   "\nConf:" + std::to_string(static_cast<int>(person.confidence * 100)) + "%";

			marker_array->markers.push_back(text_marker);
		}

		m_pubTrackingMarkers->publish(std::move(marker_array));
	}

};

}  // namespace laser_people_detector

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(laser_people_detector::LaserModelHostFollower)
