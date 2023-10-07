import math
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class Intersection(object):
    def __init__(self, point, rect):
        self.point = point
        self.rect = rect


class RectSide(object):
    def __init__(self, line):
        self.side = line
        self.visited = False


class RectPlane(object):
    def __init__(self, points, rect):
        self.points = points
        self.rect = rect


class Rectangular(object):
    def __init__(self, center: np.ndarray, length: float, width: float, height: float):
        self._center = center
        self._length = length
        self._width = width
        self._height = height
        self._s = 1.0
        self.sides = []

        self.initialize_sides()

    def initialize_sides(self):
        half_length = self.length / 2
        half_width = self.width / 2
        half_height = self.height / 2

        vertices = [
            (self.center[0] + half_length, self.center[1] + half_width, self.center[2] + half_height),
            (self.center[0] + half_length, self.center[1] + half_width, self.center[2] - half_height),
            (self.center[0] + half_length, self.center[1] - half_width, self.center[2] + half_height),
            (self.center[0] + half_length, self.center[1] - half_width, self.center[2] - half_height),
            (self.center[0] - half_length, self.center[1] + half_width, self.center[2] + half_height),
            (self.center[0] - half_length, self.center[1] + half_width, self.center[2] - half_height),
            (self.center[0] - half_length, self.center[1] - half_width, self.center[2] + half_height),
            (self.center[0] - half_length, self.center[1] - half_width, self.center[2] - half_height)
        ]

        edges = [
            [vertices[0], vertices[1]],
            [vertices[0], vertices[2]],
            [vertices[0], vertices[4]],
            [vertices[1], vertices[3]],
            [vertices[1], vertices[5]],
            [vertices[2], vertices[3]],
            [vertices[2], vertices[6]],
            [vertices[3], vertices[7]],
            [vertices[4], vertices[5]],
            [vertices[4], vertices[6]],
            [vertices[5], vertices[7]],
            [vertices[6], vertices[7]]
        ]

        sides = []
        for edge_i in edges:
            line_i = Line(edge_i[0], edge_i[1])
            side_i = RectSide(line_i)
            sides.append(side_i)

        self.sides = sides

    @property
    def center(self) -> np.ndarray:
        return np.array(self._center)

    @property
    def length(self) -> float:
        return self._length

    @property
    def width(self) -> float:
        return self._width

    @property
    def height(self) -> float:
        return self._height

    @property
    def s(self) -> float:
        return self._s

    def function(self, q: np.ndarray) -> float:
        x, y, x_0, y_0, a, b = q[0], q[1], self.center[0], self.center[1], self.width / 2, self.height / 2
        value = ((x - x_0) ** 2 + (y - y_0) ** 2 + (((x - x_0) ** 2 -
                                                     (y - y_0) ** 2 + b ** 2 - a ** 2) ** 2 + (
                                                            1 - self.s ** 2) * (
                                                            a ** 2 + b ** 2)) ** 0.5) - (a ** 2 + b ** 2)
        return value

    def compute_squicle_length_ray(self, q: np.ndarray):
        normalized_q = q / np.linalg.norm(q)
        transformed_q = np.array([normalized_q[0] / self.width, normalized_q[1] / self.height])
        normalized_transformed_q = transformed_q / np.linalg.norm(transformed_q)
        scale = math.sqrt(
            (normalized_transformed_q[0] * self.width) ** 2 + (normalized_transformed_q[1] * self.height) ** 2)
        rho_q = scale * math.sqrt(
            2 / (1 + math.sqrt(1 - 4 * self.s ** 2 * (normalized_transformed_q[0] * normalized_transformed_q[1]) ** 2)))
        return rho_q

    def check_point_inside(self, q: np.ndarray):
        x, y, z = q
        cx, cy, cz = self.center
        half_length = self.length / 2.0
        half_width = self.width / 2.0
        half_height = self.height / 2.0

        threshold = 1e-6
        if (cx - half_length - threshold <= x <= cx + half_length + threshold and
                cy - half_width - threshold <= y <= cy + half_width + threshold and
                cz - half_height - threshold <= z <= cz + half_height + threshold):
            return True
        else:
            return False

    # def plot_cube(self, ax, color='b', alpha=0.5):
    def plot_cube(self, ax, color='b', alpha=0.05):
        phi = np.arange(1, 10, 2) * np.pi / 4
        Phi, Theta = np.meshgrid(phi, phi)
        x = np.cos(Phi) * np.sin(Theta)
        y = np.sin(Phi) * np.sin(Theta)
        z = np.cos(Theta) / np.sqrt(2)
        ax.plot_surface(x * self.length + self.center[0], y * self.width + self.center[1],
                        z * self.height + self.center[2], alpha=alpha, color=color)


class Line(object):
    def __init__(self, start, end):
        self._start = start
        self._end = end

    @property
    def start(self) -> np.ndarray:
        return np.array(self._start)

    @property
    def end(self) -> np.ndarray:
        return np.array(self._end)

    def line_segment(self, param):
        return self.start + param * (self.end - self.start)

    def total_distance(self, param, point_1, point_2):
        point_on_line = self.line_segment(param)
        distance1 = np.linalg.norm(point_on_line - point_1)
        distance2 = np.linalg.norm(point_on_line - point_2)
        return distance1 + distance2

    def min_point_on_line(self, point_1, point_2):
        def total_distance(param):
            point_on_line = self.line_segment(param)
            distance1 = np.linalg.norm(point_on_line - point_1)
            distance2 = np.linalg.norm(point_on_line - point_2)
            return distance1 + distance2

        result = minimize(total_distance, 0.0, method='BFGS')
        optimal_parameter = result.x
        optimal_point = self.line_segment(optimal_parameter)
        return optimal_point

    def check_point_on_line(self, q):
        line_direction = self.end - self.start

        point_vector = q - self.start

        if np.linalg.norm(np.cross(point_vector, line_direction)) / np.linalg.norm(line_direction) < 1e-3:
            return True
        else:
            return False

    def check_point_between_line(self, q):
        start_x = self.start[0]
        start_y = self.start[1]
        end_x = self.end[0]
        end_y = self.end[1]
        center = np.array([(start_x + end_x) / 2, (start_y + end_y) / 2])
        width = abs(start_x - end_x)
        height = abs(start_y - end_y)
        virtual_rect = Rectangular(center, width, height)
        value = virtual_rect.function(q)
        if value < 1e-6:
            return True
        else:
            return False

    def plot_line(self, ax):
        x_set = [self.start[0], self.end[0]]
        y_set = [self.start[1], self.end[1]]
        ax.plot(x_set, y_set, color='r', linewidth=1.5)
        return ax

    def distance_point_to_line(self, point):
        point1 = self.start
        point2 = self.end

        line_direction = point2 - point1

        point_vector = point - point1

        distance = np.linalg.norm(np.cross(point_vector, line_direction)) / np.linalg.norm(line_direction)

        return distance


class BugPlanner(object):

    def __init__(self, start_point, goal_point, step_size, inflated_size, obstacle_list):
        self.start_point = start_point
        self.goal_point = goal_point
        self.step_size = step_size
        self.inflated_size = inflated_size
        self.obstacle_list = obstacle_list
        self.current_start_point = start_point
        self.path = [start_point]
        self.inflated_rects = []

        self.min_intersection = None
        self.min_obstacle = None
        self.nearest_rect_side_point = None
        self.distance_from_start_to_side = None
        self.nearest_rect_plane = None
        self.all_intersections = None

        self.initialize_obstacle()
        # print(self.obstacles)

    def initialize_obstacle(self):
        for obs_data_i in self.obstacle_list:
            center_i = np.array(obs_data_i[0])
            length_i = obs_data_i[1] + 2 * self.inflated_size
            width_i = obs_data_i[2] + 2 * self.inflated_size
            height_i = obs_data_i[3] + 2 * self.inflated_size
            obs_i = Rectangular(center_i, length_i, width_i, height_i)
            self.inflated_rects.append(obs_i)

    @staticmethod
    def distance(point_1, point_2):
        return np.linalg.norm(point_1 - point_2)

    @staticmethod
    def check_point_in_rect_side(point, rect):
        for side in rect.sides:
            line = side.side
            if line.distance_point_to_line(point) < 1e-3:
                # print("point_to_line", line.distance_point_to_line(point))
                return True
        else:
            return False

    @staticmethod
    def find_line_line_distance(line_1, line_2):
        p1 = np.array(line_1.start)
        q1 = np.array(line_1.end)
        p2 = np.array(line_2.start)
        q2 = np.array(line_2.end)

        v1 = q1 - p1
        v2 = q2 - p2
        w0 = p1 - p2

        a = np.dot(v1, v1)
        b = np.dot(v1, v2)
        c = np.dot(v2, v2)
        d = np.dot(v1, w0)
        e = np.dot(v2, w0)

        denominator = a * c - b * b

        if denominator == 0:
            t = 0
            s = e / c if c > 0 else 0
        else:
            t = (b * e - c * d) / denominator
            s = (a * e - b * d) / denominator

        closest_point1 = p1 + t * v1
        closest_point2 = p2 + s * v2

        distance = np.linalg.norm(closest_point1 - closest_point2)
        return distance

    def line_rectangle_intersection(self, line, rect, collision_with_line=True):
        start = line.start
        end = line.end
        box_center = rect.center
        box_size = [rect.length, rect.width, rect.height]

        half_size = [s / 2 for s in box_size]

        box_min = [box_center[i] - half_size[i] for i in range(3)]
        box_max = [box_center[i] + half_size[i] for i in range(3)]

        t_near = -float('inf')
        t_far = float('inf')

        for i in range(3):
            if start[i] == end[i]:
                if start[i] < box_min[i] or start[i] > box_max[i]:
                    return False, None
            else:
                t1 = (box_min[i] - start[i]) / (end[i] - start[i])
                t2 = (box_max[i] - start[i]) / (end[i] - start[i])

                if t1 > t2:
                    t1, t2 = t2, t1
                if t1 > t_near:
                    t_near = t1
                if t2 < t_far:
                    t_far = t2
                if t_near > t_far:
                    return False, None

        if t_near > 1 or t_far < 0:
            return False, None

        all_intersections = []
        intersection_start = [start[i] + t_near * (end[i] - start[i]) for i in range(3)]
        intersection_end = [start[i] + t_far * (end[i] - start[i]) for i in range(3)]

        if collision_with_line:
            if self.distance(np.array(intersection_start), np.array(intersection_end)) < 1e-3:
                intersection = Intersection(intersection_start, rect)
                all_intersections.append(intersection)
            else:
                intersection = Intersection(intersection_start, rect)
                all_intersections.append(intersection)

                intersection = Intersection(intersection_end, rect)
                all_intersections.append(intersection)
        else:
            if not self.check_point_in_rect_side(intersection_start, rect):
                intersection = Intersection(intersection_start, rect)
                all_intersections.append(intersection)
            if not self.check_point_in_rect_side(intersection_end, rect):
                intersection = Intersection(intersection_end, rect)
                all_intersections.append(intersection)

        # print("all_intersections", all_intersections)

        if len(all_intersections) == 0:
            return False, None
        else:
            return True, all_intersections

    def check_line_all_obstacles_intersection(self, line):
        for rect_i in self.inflated_rects:
            # print("line", line.start, line.end)
            intersection, intersections = self.line_rectangle_intersection(line, rect_i)
            inside = rect_i.check_point_inside(line.start) or rect_i.check_point_inside(line.end)
            # if (intersection and not inside) or len(intersections) > 1:
            #     # print("rect center", rect_i.center)
            #     return True
        return True

    def nearest_intersection(self):
        all_intersections = []
        line = Line(self.current_start_point, self.goal_point)

        for rect_i in self.inflated_rects:
            if self.min_obstacle is not None:
                if self.distance(rect_i.center, self.min_obstacle.center) < 1e-3:
                    continue
            intersect, intersections = self.line_rectangle_intersection(line, rect_i)
            if intersect:
                if len(intersections) == 1:
                    all_intersections.append(intersections[0])
                else:
                    all_intersections.append(intersections[0])
                    all_intersections.append(intersections[1])
        if len(all_intersections) == 0:
            self.min_intersection = None
            return
        min_distance_to_start = np.inf
        min_intersection_to_start = None

        for intersection_i in all_intersections:
            distance_to_start = self.distance(intersection_i.point, self.current_start_point)
            if distance_to_start < min_distance_to_start:
                min_distance_to_start = distance_to_start
                min_intersection_to_start = intersection_i
        self.min_intersection = min_intersection_to_start
        self.all_intersections = all_intersections
        # print(self.min_intersection.point)

    def step_toward_intersection(self):
        intersection_to_start = self.min_intersection
        distance_to_start = self.distance(self.current_start_point, intersection_to_start.point)
        step_num = int(distance_to_start / self.step_size)
        if distance_to_start < 1e-5:
            distance_to_start = 1e-5
        vector_to_intersection = (intersection_to_start.point - self.current_start_point) / distance_to_start
        for step_i in range(step_num):
            self.current_start_point = self.current_start_point + vector_to_intersection * self.step_size
            self.path.append(self.current_start_point)
        self.current_start_point = intersection_to_start.point
        self.path.append(intersection_to_start.point)

    def step_toward_side(self):
        nearest_side_point = self.nearest_rect_side_point
        distance_to_corner = self.distance(self.current_start_point, nearest_side_point)
        step_num = int(distance_to_corner / self.step_size)
        if distance_to_corner < 1e-5:
            distance_to_corner = 1e-5
        vector_to_corner = (nearest_side_point - self.current_start_point) / distance_to_corner
        for step_i in range(step_num):
            self.current_start_point = self.current_start_point + vector_to_corner * self.step_size
            self.path.append(self.current_start_point)
        self.current_start_point = nearest_side_point
        self.path.append(nearest_side_point)

    def step_toward_goal(self):
        distance_to_start = self.distance(self.current_start_point, self.goal_point)
        step_num = int(distance_to_start / self.step_size)
        if distance_to_start < 1e-5:
            distance_to_start = 1e-5
        vector_to_intersection = (self.goal_point - self.current_start_point) / distance_to_start
        for step_i in range(step_num):
            self.current_start_point = self.current_start_point + vector_to_intersection * self.step_size
            self.path.append(self.current_start_point)
        self.current_start_point = self.goal_point
        self.path.append(self.goal_point)

    def nearest_obstacle(self):
        self.min_obstacle = self.min_intersection.rect
        # print(self.min_obstacle.center)

    def find_intersection_nearest_side(self):
        nearest_obstacle = self.min_obstacle
        nearest_intersection_point = self.min_intersection.point
        # print("nearest_intersection_point", nearest_intersection_point)

        distance_from_start_to_side = np.inf
        cost_to_go = np.inf
        nearest_point_on_line = None

        for side in nearest_obstacle.sides:
            # print("current_point", self.current_start_point)
            # print("corner_point", corner.corner)
            # print("corner_visited", corner.visited)
            side_line = side.side
            # print(abs(nearest_intersection_point[0] - (side_line.start[0] + side_line.end[0]) / 2))
            # print(abs(nearest_intersection_point[1] - (side_line.start[1] + side_line.end[1]) / 2))
            # print(abs(nearest_intersection_point[2] - (side_line.start[2] + side_line.end[2]) / 2))
            if (abs(nearest_intersection_point[0] - (side_line.start[0] + side_line.end[0]) / 2) < 1e-3 or
                    abs(nearest_intersection_point[1] - (side_line.start[1] + side_line.end[1]) / 2) < 1e-3 or
                    abs(nearest_intersection_point[2] - (side_line.start[2] + side_line.end[2]) / 2) < 1e-3):
                point_on_line = side_line.min_point_on_line(nearest_intersection_point, self.goal_point)
                if self.distance(nearest_intersection_point, point_on_line) + \
                        self.distance(point_on_line, self.goal_point) < cost_to_go:
                    nearest_point_on_line = point_on_line
                    distance_from_start_to_side = self.distance(self.current_start_point, point_on_line)
                    cost_to_go = self.distance(nearest_intersection_point, point_on_line) + \
                                 self.distance(point_on_line, self.goal_point)
                    # print("corner", corner.corner)
                    # print("current_point", self.current_start_point)
                    # print("distance_from_start_to_corner", distance_from_start_to_corner)
        self.nearest_rect_side_point = nearest_point_on_line
        self.distance_from_start_to_side = distance_from_start_to_side
        # print("nearest_corner", self.nearest_rect_corner)

    def find_nearest_side(self):
        nearest_obstacle = self.min_obstacle
        nearest_intersection_point = self.min_intersection.point

        distance_from_start_to_side = np.inf
        cost_to_go = np.inf
        nearest_rect_side = None
        nearest_point_on_line = None

        # print("\n=============================")
        # print("self.current_start_point", self.current_start_point)
        for side in nearest_obstacle.sides:
            # print("current_point", self.current_start_point)
            # print("corner_point", corner.corner)
            # print("corner_visited", corner.visited)
            side_line = side.side
            point_on_line = side_line.min_point_on_line(nearest_intersection_point, self.goal_point)
            # if not side.visited and (
            #         abs(self.current_start_point[0] - (side_line.start[0] + side_line.end[0]) / 2) < 1e-3 or
            #         abs(self.current_start_point[1] - (side_line.start[1] + side_line.end[1]) / 2) < 1e-3 or
            #         abs(self.current_start_point[2] - (side_line.start[2] + side_line.end[2]) / 2) < 1e-3):
            if not side.visited and (
                    abs(self.current_start_point[0] - point_on_line[0]) < 1e-3 or
                    abs(self.current_start_point[1] - point_on_line[1]) < 1e-3 or
                    abs(self.current_start_point[2] - point_on_line[2]) < 1e-3):
                # point_on_line = side_line.min_point_on_line(nearest_intersection_point, self.goal_point)
                if self.distance(self.current_start_point, point_on_line) + \
                        self.distance(point_on_line, self.goal_point) < cost_to_go:
                    nearest_rect_side = side
                    nearest_point_on_line = point_on_line
                    distance_from_start_to_side = self.distance(self.current_start_point, point_on_line)
                    cost_to_go = self.distance(self.current_start_point, point_on_line) + \
                                 self.distance(point_on_line, self.goal_point)
                    # print("corner", corner.corner)
                    # print("current_point", self.current_start_point)
                    # print("distance_from_start_to_corner", distance_from_start_to_corner)
        # print("point_on_line", point_on_line)
        for side in nearest_obstacle.sides:
            if (self.distance(nearest_rect_side.side.start, side.side.start) < 1e-3 and
                self.distance(nearest_rect_side.side.end, side.side.end) < 1e-3) or \
                    (self.distance(nearest_rect_side.side.start, side.side.end) < 1e-3 and
                     self.distance(nearest_rect_side.side.end, side.side.start) < 1e-3):
                # print("=================visited==============")
                side.visited = True
        self.nearest_rect_side_point = nearest_point_on_line
        self.distance_from_start_to_side = distance_from_start_to_side
        # print("nearest_corner", self.nearest_rect_corner)

    def one_step_along_rect(self):
        nearest_rect_side = self.nearest_rect_side_point
        distance_from_start_to_side_point = self.distance(self.current_start_point, nearest_rect_side)
        # print(nearest_rect_corner)
        # print("distance_from_start_to_corner", self.distance_from_start_to_corner)
        if distance_from_start_to_side_point < self.step_size:
            self.current_start_point = nearest_rect_side
            self.path.append(nearest_rect_side)
        else:
            vector_to_conor = (nearest_rect_side - self.current_start_point) / distance_from_start_to_side_point
            self.current_start_point = self.current_start_point + vector_to_conor * self.step_size
            self.path.append(self.current_start_point)
        # print("current_start_point====", self.current_start_point)

    def run(self):

        while self.distance(self.current_start_point, self.goal_point) > self.step_size:
            # print(self.path[-1])
            self.nearest_intersection()
            # print("min_intersection", self.min_intersection.point)
            # print("len(self.all_intersections)", len(self.all_intersections))
            if self.min_intersection is None:
                # print("min_intersection is None")
                self.step_toward_goal()
                break
            if len(self.all_intersections) < 2:
                # print("intersections < 2")
                # print("self.current_start_point", self.current_start_point)
                # print("self.goal_point", self.goal_point)
                self.step_toward_goal()
                break
            else:
                self.nearest_obstacle()
                # print("nearest_obstacle center", self.min_obstacle.center)
                self.find_intersection_nearest_side()
                # print("self.nearest_rect_side_point", self.nearest_rect_side_point)
                if self.nearest_rect_side_point is not None:
                    line = Line(self.current_start_point, self.nearest_rect_side_point)
                    if self.check_line_all_obstacles_intersection(line):
                        self.step_toward_intersection()
                    else:
                        self.step_toward_side()
                        # print("============")

                # self.step_toward_intersection()

                line = Line(self.current_start_point, self.goal_point)
                intersection, all_intersections = self.line_rectangle_intersection(line, self.min_obstacle)
                # print(intersection)
                # print(_)
                if all_intersections is None:
                    num_intersection = 0
                else:
                    num_intersection = len(all_intersections)
                # print("num_intersections", num_intersection)
                while num_intersection > 1:
                    self.find_nearest_side()
                    # print("self.nearest_rect_side_point", self.nearest_rect_side_point)
                    while self.distance(self.current_start_point, self.nearest_rect_side_point) > self.step_size:
                        # print("distance", self.distance(self.current_start_point, self.nearest_rect_corner))
                        # print(intersection)
                        self.one_step_along_rect()
                        # print("nearest_rect_corner", self.nearest_rect_corner)
                        # print("current_start_point", self.current_start_point)
                        # print("intersection", intersection)
                        # print("intersection_points", _)
                        # print(self.current_start_point)
                    self.one_step_along_rect()
                    line = Line(self.current_start_point, self.goal_point)
                    intersection, all_intersections = self.line_rectangle_intersection(line, self.min_obstacle)
                    if all_intersections is None:
                        num_intersection = 0
                    else:
                        num_intersection = len(all_intersections)
        if self.distance(self.start_point, self.goal_point) < self.step_size:
            self.path.append(self.goal_point)
        self.smooth_path()

    def smooth_path(self):
        final_path = self.path
        new_path = [final_path[0]]
        current_point = final_path[0]
        next_point = final_path[1]
        line = Line(current_point, next_point)
        for i, path_i in enumerate(final_path):
            if line.check_point_on_line(path_i):
                continue
            else:
                new_path.append(final_path[i - 1])
                current_point = final_path[i - 1]
                next_point = path_i
                line = Line(current_point, next_point)
        new_path.append(final_path[-1])
        self.path = new_path

    def plot_cubes(self, ax):
        for rect_i in self.inflated_rects[0: len(self.obstacle_list)]:
            # rect_i.plot_cube(ax, color='b', alpha=0.2)
            rect_i.plot_cube(ax, color='b', alpha=0.02)

            origin_center_i = [rect_i.center[0], rect_i.center[1], rect_i.center[2]]
            origin_length_i = rect_i.length - 2 * self.inflated_size
            origin_width_i = rect_i.width - 2 * self.inflated_size
            origin_height_i = rect_i.height - 2 * self.inflated_size
            origin_rect_i = Rectangular(origin_center_i, origin_length_i, origin_width_i, origin_height_i)
            # origin_rect_i.plot_cube(ax, color='b', alpha=0.5)
            origin_rect_i.plot_cube(ax, color='b', alpha=0.05)

    def plot_path(self, ax):
        path_x = []
        path_y = []
        path_z = []
        for path_i in self.path:
            # print("path_i", path_i)
            path_x.append(path_i[0])
            path_y.append(path_i[1])
            path_z.append(path_i[2])
        ax.plot(path_x, path_y, path_z, '-k', linewidth=1.5)


def obstacle_adapter(obstacle_list):
    obstacle_list = [
        [
            [
                ob[0] + ob[-1] / 2,
                ob[1] + ob[-1] / 2,
                ob[2] + ob[-1] / 2,
            ],
            ob[-1],
            ob[-1],
            ob[-1],
        ]
        for ob in obstacle_list]
    return obstacle_list


if __name__ == '__main__':
    # obstacles
    # [center_x, center_y], width, height
    # obstacle_list = [[np.array([50.0, 50.0, 50.0]), 40.0, 40.0, 40.0]]
    #
    # start_point = np.array([1.0, 1.0, 1.0])
    #
    # end_point = np.array([100.0, 100.0, 100.0])

    # obstacle_list = [
    #     [[141.48373054471006, 146.55784005142408, 59.2967559003791], 43.088693800637664, 43.088693800637664,
    #      43.088693800637664],
    #     [[137.2441736890225, 82.65639648899777, 56.725557814217], 43.088693800637664, 43.088693800637664,
    #      43.088693800637664]
    #     ]
    #     obstacle_list=[[[91.1010938167906, 151.0789592228769, 86.08340591097476], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[135.2891128142748, 79.2877529763728, 91.47299758388668], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[173.5323891782093, 44.63668555537053, 162.61653155051994], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[23.652798969884124, 160.95417060385526, 138.60276777985337], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[57.42497281437761, 57.554185976735255, 91.73868526229646], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[50.83915556322631, 94.86296418308638, 150.25633586084282], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[168.9764186348353, 158.19411803676306, 166.7437658527735], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[80.76263603109288, 53.5075181119653, 31.84622032247669], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[163.7416515229149, 135.24954096248746, 72.46097609101547], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[96.31466725220228, 154.32181956559546, 169.5159291020757], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[118.16594180394186, 82.1084997124477, 147.0266557733895], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[97.98243304201627, 170.87840890244382, 27.008177381778808], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[34.2557287954926, 26.270482109634088, 160.62283962079533], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[33.12826195083585, 134.8325831041646, 43.62784998317042], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[167.78671142417843, 40.54225740301587, 24.560964154205593], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[173.94801650250471, 21.593026753462944, 82.61525311049623], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[21.82442886043523, 55.592288700681564, 35.20574099152768], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[93.58868408728603, 26.38280648364173, 155.44272583442415], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[107.70506744156137, 109.84661025910486, 25.843565299322243], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[117.93804201138343, 22.281367762442382, 97.05562443349318], 43.088693800637664, 43.088693800637664, 43.088693800637664]]

    #     path = [[138.87044982,130.37268943,21.38206579]
    # ,[136.78341471,107.20074339,81.26990471]
    # ,[135.22400714,52.57054689,98.80219346]]
    #     start_point = path[0]
    #     end_point = path[2]

    #     agent_start = [start_point]
    #     agent_end = [end_point]
    # obstacle_list  = [[[91.1010938167906, 151.0789592228769, 86.08340591097476], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[135.2891128142748, 79.2877529763728, 91.47299758388668], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[173.5323891782093, 44.63668555537053, 162.61653155051994], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[23.652798969884124, 160.95417060385526, 138.60276777985337], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[57.42497281437761, 57.554185976735255, 91.73868526229646], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[50.83915556322631, 94.86296418308638, 150.25633586084282], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[168.9764186348353, 158.19411803676306, 166.7437658527735], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[80.76263603109288, 53.5075181119653, 31.84622032247669], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[163.7416515229149, 135.24954096248746, 72.46097609101547], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[96.31466725220228, 154.32181956559546, 169.5159291020757], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[118.16594180394186, 82.1084997124477, 147.0266557733895], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[97.98243304201627, 170.87840890244382, 27.008177381778808], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[34.2557287954926, 26.270482109634088, 160.62283962079533], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[33.12826195083585, 134.8325831041646, 43.62784998317042], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[167.78671142417843, 40.54225740301587, 24.560964154205593], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[173.94801650250471, 21.593026753462944, 82.61525311049623], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[21.82442886043523, 55.592288700681564, 35.20574099152768], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[93.58868408728603, 26.38280648364173, 155.44272583442415], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[107.70506744156137, 109.84661025910486, 25.843565299322243], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[117.93804201138343, 22.281367762442382, 97.05562443349318], 43.088693800637664, 43.088693800637664, 43.088693800637664]]
    # start_point    = [169.5660706245593, 42.19868941825102, 126.5607928062335]
    # end_point      = [140.23561876825414, 22.92378200868626, 35.22700017229256]
    # obstacle_list  = [[[57.45655634249066, 123.10847131127363, 38.66631841141745], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[101.2840141170797, 73.52053122209, 105.38707892112028], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[167.87482715638524, 31.267841169372296, 177.23332814466355], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[26.1108108952177, 23.529291075567805, 116.36729150353237], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[108.24273552218702, 168.06528875337176, 101.49413287082005], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[81.57153676771192, 57.5072464579945, 37.04246964638349], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[170.36007682005248, 162.20367719070552, 48.47346055691678], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[147.69084884637533, 168.7979227135558, 172.7495418054965], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[95.258337018351, 89.68578979799268, 171.51435098075052], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[36.18051868050385, 87.61905852298253, 168.95889626133823], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[176.23663385548866, 50.17696860021681, 66.86973162672786], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[30.77567091410083, 152.71055169574493, 147.0738471074391], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[163.71514432348866, 104.57373656382956, 158.63744785694263], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[91.72707179305992, 177.82232240238085, 160.34589689895216], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[168.64246211040899, 161.97702270457117, 113.24633600730408], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[25.013906064696883, 23.00286455457447, 41.242630902166965], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[37.43672574741474, 93.81889141428869, 96.06588943271004], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[114.1077458106423, 156.79408799586412, 23.219259299107776], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[27.62293449371972, 178.20541379947267, 33.292374431148275], 43.088693800637664, 43.088693800637664, 43.088693800637664], [[170.79942554185325, 106.64611515241407, 22.312008896489427], 43.088693800637664, 43.088693800637664, 43.088693800637664]]
    # start_point    = [144.08043064681382, 183.1402207422691, 34.04406197356165]
    # end_point      = [125.3867096234803, 6.122766418403282, 156.45990529121053]
    obstacle_list = [
        [[136.90321135971865, 103.7409111509183, 33.467157544968714],
         47.425244059867495, 47.425244059867495, 47.425244059867495],
        [[51.03130516097224, 93.1029516790268, 162.41304774035635], 47.425244059867495, 47.425244059867495,
         47.425244059867495],
        [[171.11597754095536, 148.37427994689713, 103.60512018410728], 47.425244059867495, 47.425244059867495,
         47.425244059867495],
        [[61.67067938738376, 41.95425941327129, 88.17425009930668], 47.425244059867495, 47.425244059867495,
         47.425244059867495],
        [[171.62434117619506, 32.63469955095063, 88.39295445862425], 47.425244059867495, 47.425244059867495,
         47.425244059867495],
        [[97.00177089277886, 166.928179567834, 57.287532263613414], 47.425244059867495, 47.425244059867495,
         47.425244059867495],
        [[103.08692296044848, 168.7450503596725, 147.13184931648127], 47.425244059867495, 47.425244059867495,
         47.425244059867495],
        [[40.62603614579494, 26.197525781239726, 164.42320334126663], 47.425244059867495, 47.425244059867495,
         47.425244059867495],
        [[121.1391699202253, 37.85967345106205, 165.9833946601597], 47.425244059867495, 47.425244059867495,
         47.425244059867495],
        [[25.031523208625153, 166.77014903922358, 66.6495275686932], 47.425244059867495, 47.425244059867495,
         47.425244059867495],
        [[36.068586243863734, 167.18942579246092, 158.92929340632188], 47.425244059867495, 47.425244059867495,
         47.425244059867495],
        [[169.09532438883093, 175.05549381441247, 167.5702441316549], 47.425244059867495, 47.425244059867495,
         47.425244059867495],
        [[173.03225293773164, 105.28749720267533, 171.79604489584938], 47.425244059867495, 47.425244059867495,
         47.425244059867495],
        [[112.23333405124836, 30.79437569370776, 25.209624226275217], 47.425244059867495, 47.425244059867495,
         47.425244059867495],
        [[45.424268151495355, 32.46869447169, 25.549726587997252], 47.425244059867495, 47.425244059867495,
         47.425244059867495]]
    start_point = [80.51281706004995, 165.9302437112153, 176.32096510189072]
    end_point = [20.46175513947782, 35.333952441221776, 132.31281878503614]
    agent_start = [start_point]
    agent_end = [end_point]

    step_size = 5.0
    inflated_size = 4.0

    # TODO： 写一个过滤障碍物的，  还没完成
    obstacle_list = [
        ob
        for ob in obstacle_list
        if (
                np.all(np.array(ob[0]) - np.array(ob[1:]) / 2 < np.max([start_point, end_point], axis=0))
                and
                np.all(np.array(ob[0]) + np.array(ob[1:]) / 2 > np.min([start_point, end_point], axis=0))
        )
    ]
    inflate_obs_min_max = [
        (np.array(ob[0]) - np.array(ob[1:]) / 2 - inflated_size,
         np.array(ob[0]) + np.array(ob[1:]) / 2 + inflated_size
         )
        for ob in obstacle_list
    ]
    for minp, maxp in inflate_obs_min_max:
        if np.all(minp < start_point) and np.all(start_point < maxp):
            print('start_point')
            print(start_point)
            print('minp,maxp')
            print(minp)
            print(maxp)
        if np.all(minp < end_point) and np.all(end_point < maxp):
            print('start_point')
            print(end_point)
            print('minp,maxp')
            print(minp)
            print(maxp)
    # exit()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([0, 200])
    ax.set_ylim([0, 200])
    ax.set_zlim([0, 200])
    ax.set_aspect('equal')

    for i, agent_start_i in enumerate(agent_start):
        start_point = np.array(agent_start[i])
        end_point = np.array(agent_end[i])

        # print("=============agent", i, "===========")
        # print("start_point", start_point)
        # print("end_point", end_point)
        time_start = time.time()
        bug_planner = BugPlanner(start_point, end_point, step_size, inflated_size, obstacle_list)

        bug_planner.run()
        final_path = bug_planner.path
        bug_planner.plot_path(ax)
        time_end = time.time()
        total_time = time_end - time_start
        print(total_time)
        ax.plot(*start_point, 'r*')
        ax.plot(*end_point, 'go')
    bug_planner = BugPlanner(start_point, end_point, step_size, inflated_size, obstacle_list)
    bug_planner.plot_cubes(ax)

    plt.show()