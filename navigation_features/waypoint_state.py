WAYPOINT_SIGNATURE_ATTR = "_social_safety_reward_waypoint_signature"
CURRENT_WAYPOINTS_ATTR = "_social_safety_reward_current_waypoints"
LAST_REACHED_WAYPOINT_ATTR = "_social_safety_reward_last_reached_waypoint_index"
WAYPOINT_ADVANCE_RADIUS_ATTR = "_social_safety_reward_waypoint_advance_radius"
PROGRESS_TARGET_SIGNATURE_ATTR = "_social_safety_reward_progress_target_signature"
PROGRESS_TARGET_DISTANCE_ATTR = "_social_safety_reward_progress_target_distance"


def advance_waypoint_index(robot_x, robot_y, waypoints, last_reached_index, advance_radius):
    """Advance reached index by radius hits or by clearly passing a waypoint."""
    last_reached_index = int(last_reached_index)
    reached_hits = 0
    next_index = last_reached_index + 1

    while next_index < len(waypoints):
        waypoint = waypoints[next_index]
        distance = ((float(waypoint[0]) - robot_x) ** 2 + (float(waypoint[1]) - robot_y) ** 2) ** 0.5
        if distance <= advance_radius:
            reached_hits += 1
        elif not waypoint_is_passed(robot_x, robot_y, waypoints, next_index, advance_radius):
            break

        last_reached_index = next_index
        next_index += 1

    return last_reached_index, reached_hits


def waypoint_is_passed(robot_x, robot_y, waypoints, index, tolerance):
    if len(waypoints) < 2:
        return False

    waypoint = waypoints[index]
    if index > 0:
        anchor = waypoints[index - 1]
        direction = (float(waypoint[0]) - float(anchor[0]), float(waypoint[1]) - float(anchor[1]))
    else:
        next_waypoint = waypoints[index + 1]
        direction = (float(next_waypoint[0]) - float(waypoint[0]), float(next_waypoint[1]) - float(waypoint[1]))

    length = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
    if length <= 1e-8:
        return False

    unit = (direction[0] / length, direction[1] / length)
    robot_offset = (float(robot_x) - float(waypoint[0]), float(robot_y) - float(waypoint[1]))
    return robot_offset[0] * unit[0] + robot_offset[1] * unit[1] > float(tolerance)
