import numpy as np

from environment_inspection.episode_analysis import collect_analysis_episodes


def run_failure_analysis(env, policy, config):
    cfg = config["failure_analysis"]
    base_seed = int(cfg["seed"])
    episodes = int(cfg["episodes"])
    max_steps = int(cfg["max_steps"])
    threshold = float(cfg["clearance_threshold"])

    failed_clearances = []
    print(
        "Failure analysis started: "
        f"policy={config['policy']['type']} seed={base_seed} episodes={episodes} "
        f"max_steps={max_steps} threshold={threshold:.3f}m"
    )

    collection = collect_analysis_episodes(
        env,
        policy,
        base_seed=base_seed,
        episode_count=episodes,
        max_steps=max_steps,
        start_trace=initial_spawn_clearance,
    )
    for episode in collection.episodes:
        if not bool(episode.final_info.get("SUCCESS", False)):
            failed_clearances.append(
                {
                    "seed": episode.seed,
                    "clearance": episode.trace["clearance"],
                    "entity_type": episode.trace["entity_type"],
                    "failure_type": failure_type(
                        episode.final_info,
                        episode.terminated,
                        episode.truncated,
                    ),
                }
            )

    print_failure_clearance_summary(
        failed_clearances,
        completed=len(collection.episodes),
        requested=episodes,
        skipped_resets=collection.skipped_resets,
        threshold=threshold,
    )


def initial_spawn_clearance(env):
    base_env = env.unwrapped
    robot = getattr(base_env, "robot", None)
    if robot is None:
        return {"clearance": float("inf"), "entity_type": "none"}

    robot_geom = entity_geometry(robot)
    best = {"clearance": float("inf"), "entity_type": "none"}
    for entity in spawn_entities(base_env):
        geom = entity_geometry(entity)
        if geom is None:
            continue
        clearance = float(robot_geom.distance(geom))
        if clearance < best["clearance"]:
            best = {"clearance": clearance, "entity_type": getattr(entity, "name", type(entity).__name__)}
    return best


def spawn_entities(base_env):
    entities = []
    entities.extend(getattr(base_env, "static_humans", []))
    entities.extend(getattr(base_env, "dynamic_humans", []))
    entities.extend(getattr(base_env, "plants", []))
    entities.extend(getattr(base_env, "tables", []))
    entities.extend(getattr(base_env, "laptops", []))
    entities.extend(getattr(base_env, "walls", []))

    for interaction in getattr(base_env, "moving_interactions", []) + getattr(base_env, "static_interactions", []):
        entities.extend(getattr(interaction, "humans", []))
    for interaction in getattr(base_env, "h_l_interactions", []):
        entities.append(getattr(interaction, "human", None))
        entities.append(getattr(interaction, "laptop", None))

    seen = set()
    unique = []
    for entity in entities:
        if entity is None or getattr(entity, "name", None) == "robot":
            continue
        marker = id(entity)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(entity)
    return unique


def entity_geometry(entity):
    from shapely.geometry import Point, Polygon
    from socnavgym.envs.utils.utils import get_coordinates_of_rotated_rectangle

    name = getattr(entity, "name", None)
    if name in ("robot", "plant"):
        return Point((entity.x, entity.y)).buffer(float(entity.radius))
    if name == "human":
        return Point((entity.x, entity.y)).buffer(float(entity.width) / 2.0)
    if name in ("laptop", "table"):
        return Polygon(get_coordinates_of_rotated_rectangle(entity.x, entity.y, entity.orientation, entity.length, entity.width))
    if name == "wall":
        return Polygon(get_coordinates_of_rotated_rectangle(entity.x, entity.y, entity.orientation, entity.length, entity.thickness))
    return None


def failure_type(info, terminated, truncated):
    if info.get("COLLISION_HUMAN"):
        return "collision_human"
    if info.get("COLLISION_WALL"):
        return "collision_wall"
    if info.get("COLLISION_OBJECT"):
        return "collision_object"
    if info.get("COLLISION"):
        return "collision"
    if info.get("TIMEOUT") or truncated:
        return "timeout"
    if terminated:
        return "terminated"
    return "max_steps"


def print_failure_clearance_summary(failed_clearances, completed, requested, skipped_resets, threshold):
    print("\nFailure clearance summary")
    print(f"  requested_episodes: {requested}")
    print(f"  completed_episodes: {completed}")
    print(f"  skipped_reset_failures: {skipped_resets}")
    print(f"  failed_episodes: {len(failed_clearances)}")
    print(f"  clearance_threshold_m: {threshold:.6f}")

    if not failed_clearances:
        print("  no failed episodes recorded.")
        return

    values = np.array([entry["clearance"] for entry in failed_clearances], dtype=np.float32)
    below = values < threshold
    print(f"  min_failed_clearance_m: {float(np.min(values)):.6f}")
    print(f"  max_failed_clearance_m: {float(np.max(values)):.6f}")
    print(f"  mean_failed_clearance_m: {float(np.mean(values)):.6f}")
    print(f"  median_failed_clearance_m: {float(np.median(values)):.6f}")
    print(f"  failures_below_threshold: {int(np.sum(below))}/{len(values)} = {float(np.mean(below)):.2%}")

    print("  failed_episode_details:")
    for entry in failed_clearances:
        print(
            f"    seed={entry['seed']} failure={entry['failure_type']} "
            f"nearest_entity={entry['entity_type']} clearance_m={entry['clearance']:.6f}"
        )
