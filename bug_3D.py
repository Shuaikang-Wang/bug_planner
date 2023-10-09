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
        # print("=====nearest_intersection_point", nearest_intersection_point)
        # print("=====self.current_start_point", self.current_start_point)

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
            point_on_line = side_line.min_point_on_line(self.current_start_point, self.goal_point)
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
            # print("len(self.all_intersections)", len(self.all_intersections))
            if self.min_intersection is None:
                # print("min_intersection is None")
                # print("self.current_start_point", self.current_start_point)
                self.step_toward_goal()
                # print("==========step_toward_goal===========")
                break
            # print("min_intersection", self.min_intersection.point)
            # print("num all_intersections", len(self.all_intersections))
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
                # if self.nearest_rect_side_point is not None:
                #     line = Line(self.current_start_point, self.nearest_rect_side_point)
                #     if self.check_line_all_obstacles_intersection(line):
                #         self.step_toward_intersection()
                #     else:
                #         self.step_toward_side()
                        # print("============")

                self.step_toward_intersection()

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
    obstacle_list  = [[[213.9283373311857, 193.89630506036022, 38.490399920953315], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[226.2619564244664, 115.3390351474343, 72.4239768045762], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[106.31690747755499, 94.04341081849003, 180.18636757745918], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[111.44334882165955, 212.49129189095098, 179.60926260404193], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[84.68037819852248, 58.59453149118684, 74.24937580518196], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[58.47970283329556, 31.98446406175574, 202.5816008940011], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[200.63439308156177, 58.40215586899331, 93.26370573966383], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[84.28499253497542, 159.05790782926286, 165.6280279188394], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[149.0251839940742, 184.37980211737607, 86.24495944720239], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[83.95645293182133, 147.23394328752698, 21.149163448461756], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[66.83697379806426, 192.05274355783683, 223.51255776274772], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[220.48403255055922, 46.98672158780225, 39.91864377208721], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[162.25060024977918, 28.715908411577303, 174.0324229036533], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[202.9647336949682, 228.30418319588787, 147.0557383079105], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[99.5578925603359, 41.92218381347375, 142.80557985888805], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[173.71391575440208, 176.99072175257163, 152.22216771507092], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[92.55194535249821, 149.42930183652726, 108.73980162724509], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[79.2613092938914, 91.62260744918702, 123.16404816497347], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[151.8368587781627, 186.8935275508918, 18.093023995566096], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[222.72789033062054, 137.83882386796284, 181.67790016324085], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[72.6031509406197, 211.41192384219062, 100.69993523426623], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[152.59063821175056, 113.90740918103805, 80.94700295859778], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[209.95401734405357, 44.973287806224036, 153.03375344226387], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[27.162500279962543, 130.8737835187746, 55.73378549692969], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[193.02103155901068, 93.87201308500767, 25.17740095822725], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[171.2195380631111, 119.36318245558567, 184.82562313006284], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[29.712603229423443, 37.50660418672746, 53.749969914927874], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[165.2540647550945, 56.9355233176037, 227.82915082878563], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[39.74321198188467, 176.4058078378684, 151.46964039346756], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[31.423202460222246, 121.29850936726297, 199.46746894481612], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[62.35831516400338, 213.68781756591423, 50.25676592097813], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[201.63409056554815, 181.22804837466805, 89.13212489347605], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[76.81301526275831, 88.55727527317687, 228.8537311810865], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[163.32961870756048, 17.679089392513426, 21.445287423258705], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[24.93445028465937, 210.80057820937353, 103.82226723283617], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[173.130610360741, 209.4652282687426, 223.65427662393049], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[211.78739897195402, 124.0105394585482, 131.59784629306736], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[38.774322759278746, 224.14201723201185, 171.7540753767732], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[47.85114201356731, 32.95149490282377, 133.85300271711776], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[18.65062551494124, 79.10621367538249, 110.19502518629646], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[73.26220067250792, 88.9582479964374, 23.145944979497465], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[155.9405953109963, 223.88216394401678, 151.5636865501135], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[111.72311253981498, 158.7485770432011, 216.77858319947376], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[140.59474403809347, 44.73835131116182, 69.73844255913241], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[227.74693110742226, 75.56219238212272, 201.95214255295267], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[148.3701322717032, 64.47431571390659, 23.78011670327308], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[150.6925278893587, 231.97456761850563, 21.906048481323566], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[111.37155178780577, 26.615806152746497, 192.5919049047729], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[219.87506576912858, 197.1214770710151, 209.1848766024535], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[140.99219544388524, 134.3816308218976, 21.60014934240143], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[21.2940796397438, 224.39612659451626, 228.9753523781225], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[183.61362960945283, 146.43422406581664, 230.8308278174008], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[180.4536708025949, 230.20484234893044, 96.63348471086421], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[32.31453724177375, 143.10043305293436, 102.10389081126326], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[155.76703417699235, 27.81435025578824, 120.39122232645835], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[228.72737245340704, 146.77997850694175, 27.698651329695515], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[91.82531251699422, 43.781843554704636, 29.343512540466218], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[131.309557659719, 229.87730158403778, 79.38565533901095], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[140.01295991196133, 107.2234426176053, 135.6068301405794], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[17.711449824796414, 199.26220810698774, 43.96665656629425], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[225.45598345502992, 183.13533580998575, 147.32348670826215], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[230.48084811636002, 19.508452101700325, 208.62992335806462], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[127.59177130168962, 222.8838895818346, 226.7052695958914], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[231.43269697397469, 229.35212487451994, 100.6057389009729], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[18.995099286268022, 83.3475046950124, 28.837660243655314], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[158.54833222131805, 101.75885198867981, 230.4390125899348], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[25.328114294498448, 76.49085091625454, 229.80202922100324], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[45.575641018842205, 76.74468486850269, 184.49560476948986], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[18.705332861048372, 178.11388435401062, 206.4640919178443], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[229.17057164962222, 125.94463844319755, 229.9142086941201], 35.35533905932738, 35.35533905932738, 35.35533905932738], [[28.36979438101544, 128.65906683743745, 149.60112520689214], 35.35533905932738, 35.35533905932738, 35.35533905932738]]
    start_point    = [156.14307248435588, 54.09101372167609, 108.29393280654631]
    end_point      = [40.527808980031736, 40.88377532265013, 19.723585527234164]
    agent_start = [start_point]
    agent_end = [end_point]
    inflated_size = 2.0

    step_size = 5.0

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
        print("bug planning time:", total_time)
        ax.plot(*start_point, 'r*')
        ax.plot(*end_point, 'go')
    bug_planner = BugPlanner(start_point, end_point, step_size, inflated_size, obstacle_list)
    bug_planner.plot_cubes(ax)

    plt.show()