"""
This A star implementation is from; https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/AStar/a_star.py

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

import math
import heapq
from collections import deque
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

show_animation = True


class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Found goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion


@dataclass(frozen=True)
class GridPath:
    """Path returned by the grid/cost-map planner."""

    cells: list
    cost: float


class CostMapAStarPlanner:
    """
    A* planner for an existing occupancy grid and optional traversal cost map.

    This class is intentionally separate from AStarPlanner so the original
    PythonRobotics-style implementation remains detachable.
    """

    def __init__(
        self,
        occupancy_grid,
        cost_map=None,
        allow_diagonal=True,
        diagonal_corner_cutting=False,
        heuristic_weight=1.0,
    ):
        self.occupancy_grid = np.asarray(occupancy_grid, dtype=bool)
        if self.occupancy_grid.ndim != 2:
            raise ValueError("occupancy_grid must be a 2D array")

        self.height, self.width = self.occupancy_grid.shape
        self.cost_map = self._prepare_cost_map(cost_map)
        self.allow_diagonal = allow_diagonal
        self.diagonal_corner_cutting = diagonal_corner_cutting
        self.heuristic_weight = heuristic_weight
        self.motion = self._motion_model(allow_diagonal)

    def _prepare_cost_map(self, cost_map):
        if cost_map is None:
            cost_map = np.ones_like(self.occupancy_grid, dtype=float)
        else:
            cost_map = np.asarray(cost_map, dtype=float)
            if cost_map.shape != self.occupancy_grid.shape:
                raise ValueError("cost_map shape must match occupancy_grid")
            cost_map = cost_map.copy()

        cost_map[self.occupancy_grid] = math.inf
        return cost_map

    def planning(self, start_cell, goal_cell):
        """
        Plan from start_cell to goal_cell.

        Args:
            start_cell: (row, col) tuple.
            goal_cell: (row, col) tuple.

        Returns:
            GridPath with cells ordered from start to goal. Empty if no path.
        """
        start = tuple(int(v) for v in start_cell)
        goal = tuple(int(v) for v in goal_cell)

        if not self._is_free(start):
            return GridPath([], math.inf)
        if not self._is_free(goal):
            return GridPath([], math.inf)

        open_heap = []
        heapq.heappush(open_heap, (self._heuristic(start, goal), 0.0, start)) # (f, g, cell)
        came_from = {}
        g_score = {start: 0.0}
        closed = set()

        while open_heap:
            _, current_cost, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                return GridPath(self._reconstruct_path(came_from, current), current_cost)

            closed.add(current)

            for dr, dc, step_cost in self.motion:
                neighbor = (current[0] + dr, current[1] + dc)
                if neighbor in closed or not self._is_free(neighbor):
                    continue
                if not self.diagonal_corner_cutting and dr != 0 and dc != 0: # (dr != 0 and dc != 0) means diagonal movement
                    # Prevent corner cutting during diagonal moves.
                    # For a diagonal step, both adjacent horizontal and vertical cells
                    # must be free; otherwise the move is rejected to avoid passing
                    # through obstacle corners.
                    if not self._is_free((current[0] + dr, current[1])):
                        continue
                    if not self._is_free((current[0], current[1] + dc)):
                        continue

                traversal_cost = self.cost_map[neighbor]
                tentative = current_cost + step_cost * traversal_cost
                if tentative < g_score.get(neighbor, math.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    priority = tentative + self._heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (priority, tentative, neighbor))

        return GridPath([], math.inf)

    def nearest_free_cell(self, cell, max_distance_cells=None):
        """Return the closest free cell to cell using grid breadth-first search."""
        start = tuple(int(v) for v in cell)
        if self._is_free(start):
            return start

        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            current, distance = queue.popleft()
            if max_distance_cells is not None and distance > max_distance_cells:
                break
            if self._is_free(current):
                return current
            for dr, dc, _ in self._motion_model(True):
                neighbor = (current[0] + dr, current[1] + dc)
                if neighbor in visited:
                    continue
                row, col = neighbor
                if 0 <= row < self.height and 0 <= col < self.width:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        return None

    def _is_free(self, cell):
        row, col = cell
        return (
            0 <= row < self.height
            and 0 <= col < self.width
            and not self.occupancy_grid[row, col]
            and np.isfinite(self.cost_map[row, col])
        )

    def _heuristic(self, cell, goal):
        return self.heuristic_weight * math.hypot(goal[0] - cell[0], goal[1] - cell[1])

    @staticmethod
    def _reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def _motion_model(allow_diagonal):
        motion = [(1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0)]
        if allow_diagonal:
            diagonal = math.sqrt(2)
            motion.extend([(1, 1, diagonal), (1, -1, diagonal), (-1, 1, diagonal), (-1, -1, diagonal)])
        return motion


def sample_path_by_distance(path, interval, include_start=True, include_goal=True):
    """
    Extract waypoints from a world-coordinate path at approximately interval metres.

    Args:
        path: sequence of (x, y) points.
        interval: waypoint spacing in metres. If <= 0, returns the input path.
    """
    if not path:
        return []
    if interval <= 0:
        return list(path)

    waypoints = [path[0]] if include_start else []
    next_distance = interval
    travelled = 0.0

    for start, end in zip(path, path[1:]):
        sx, sy = start
        ex, ey = end
        segment = math.hypot(ex - sx, ey - sy)
        if segment == 0:
            continue

        while travelled + segment >= next_distance:
            ratio = (next_distance - travelled) / segment
            waypoints.append((sx + ratio * (ex - sx), sy + ratio * (ey - sy)))
            next_distance += interval

        travelled += segment

    if include_goal and (not waypoints or waypoints[-1] != path[-1]):
        waypoints.append(path[-1])

    return waypoints


def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]
    grid_size = 2.0  # [m]
    robot_radius = 1.0  # [m]

    # set obstacle positions
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40.0)
        oy.append(60.0 - i)

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()


if __name__ == '__main__':
    main()
