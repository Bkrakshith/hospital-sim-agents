"""Names of the benchmark metrics scenario criteria may use; bench/metrics.py implements each."""
METRIC_NAMES: tuple[str, ...] = (
    "min_distance_m", "time_within_0_5m_s", "max_speed_near_people_mps", "contacts", "forbidden_area_entries",
    "unanswered_requests", "mean_response_s", "trays_correct_in_reach", "trays_total", "instruments_correct_rate",
    "instrument_mean_latency_s", "fire_out_s", "people_reached_destination_rate", "nurse_alerted_s",
    "specimen_delivered_s", "patient_left_ward_unaccompanied_s", "deterioration_alerted_s", "mean_wait_s",
    "mood_change", "rating_likeability", "rating_safety", "rating_competence", "trust_item_mean",
    "trust_delta_mean", "people_in_furniture_s", "robot_near_furniture_s", "robot_reached_destination_s",
    "fire_out_by_robot_s",
)
