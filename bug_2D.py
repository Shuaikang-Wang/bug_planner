import math
import numpy as np
import time
import copy
import matplotlib.pyplot as plt


class Intersection(object):
    def __init__(self, point, rect):
        self.point = point
        self.rect = rect


class RectCorner(object):
    def __init__(self, corner):
        self.corner = corner
        self.visited = False


class Rectangular(object):
    def __init__(self, center: np.ndarray, width: float, height: float):
        self._center = center
        self._width = width
        self._height = height
        self._s = 1.0
        self.corners = []

        self.initialize_corner()

    def initialize_corner(self):
        rect_half_width = self.width / 2
        rect_half_height = self.height / 2
        rect_left = self.center[0] - rect_half_width
        rect_right = self.center[0] + rect_half_width
        rect_top = self.center[1] + rect_half_height
        rect_bottom = self.center[1] - rect_half_height
        left_top = RectCorner(np.array([rect_left, rect_top]))
        left_bottom = RectCorner(np.array([rect_left, rect_bottom]))
        right_top = RectCorner(np.array([rect_right, rect_top]))
        right_bottom = RectCorner(np.array([rect_right, rect_bottom]))

        self.corners = [left_top,
                        left_bottom,
                        right_top,
                        right_bottom]

    @property
    def center(self) -> np.ndarray:
        return np.array(self._center)

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
        if self.function(q) < 1e-6:
            return True
        else:
            return False

    def plot_rectangular(self, ax, color='r'):
        center_rect_i = (self.center[0] - self.width / 2, self.center[1] - self.height / 2)
        original_rect_i = plt.Rectangle(center_rect_i, self.width, self.height, edgecolor=color,
                                        facecolor=color, linewidth=1.5, fill=True, alpha=0.6)
        ax.add_patch(original_rect_i)
        return ax


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

    def check_point_on_line(self, q):
        start_x = self.start[0]
        start_y = self.start[1]
        end_x = self.end[0]
        end_y = self.end[1]
        value = abs((end_y - start_y) * q[0] - (end_x - start_x) * q[1] + end_x * start_y - end_y * start_x)
        if value < 1e-3:
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
        self.nearest_rect_corner = None
        self.distance_from_start_to_corner = None
        self.in_inflated_obs = 0

        # self.recalculate_inflated_size()
        self.initialize_obstacle()
        # print(self.obstacles)

    def recalculate_inflated_size(self):
        shortest_distance = np.inf
        normal_vector = np.array([])
        for rect in self.obstacle_list:
            center_x, center_y = rect[0]
            width = rect[1]
            height = rect[2]
            rect_half_width = width / 2
            rect_half_height = height / 2
            rect_left = center_x - rect_half_width
            rect_right = center_x + rect_half_width
            rect_top = center_y + rect_half_height
            rect_bottom = center_y - rect_half_height
            left_top = np.array([rect_left, rect_top])
            left_bottom = np.array([rect_left, rect_bottom])
            right_top = np.array([rect_right, rect_top])
            right_bottom = np.array([rect_right, rect_bottom])
            corners = [left_top,
                       left_bottom,
                       right_bottom,
                       right_top]
            for i in range(4):
                x1, y1 = corners[i]
                x2, y2 = corners[(i + 1) % 4]
                dx = x2 - x1
                dy = y2 - y1

                vx = self.current_start_point[0] - x1
                vy = self.current_start_point[1] - y1
                distance = abs(vx * dy - vy * dx) / math.sqrt(dx ** 2 + dy ** 2)

                dot_product = vx * dx + vy * dy
                if 0 <= dot_product <= dx ** 2 + dy ** 2:
                    pass
                else:
                    distance = min(
                        math.sqrt((self.current_start_point[0] - x1) ** 2 + (self.current_start_point[1] - y1) ** 2),
                        math.sqrt((self.current_start_point[0] - x2) ** 2 + (self.current_start_point[1] - y2) ** 2)
                    )
                if distance < shortest_distance:
                    shortest_distance = distance
        # print("shortest_distance", shortest_distance)
        if shortest_distance < self.inflated_size:
            self.in_inflated_obs = 1

    def initialize_obstacle(self):
        for obs_data_i in self.obstacle_list:
            center_i = np.array(obs_data_i[0])
            width_i = obs_data_i[1] + 2 * self.inflated_size
            height_i = obs_data_i[2] + 2 * self.inflated_size
            obs_i = Rectangular(center_i, width_i, height_i)
            self.inflated_rects.append(obs_i)

    def distance(self, point_1, point_2):
        return np.linalg.norm(point_1 - point_2)

    def check_point_in_rect_corner(self, point, rect):
        for corner in rect.corners:
            if self.distance(corner.corner, point) < 1e-3:
                return True
        else:
            return False

    def line_rectangle_intersection(self, line, rect):
        rect_half_width = rect.width / 2
        rect_half_height = rect.height / 2
        rect_left = rect.center[0] - rect_half_width
        rect_right = rect.center[0] + rect_half_width
        rect_top = rect.center[1] + rect_half_height
        rect_bottom = rect.center[1] - rect_half_height

        start_x = line.start[0]
        start_y = line.start[1]
        end_x = line.end[0]
        end_y = line.end[1]

        intersections = []

        if start_x == end_x:
            if rect_left <= start_x <= rect_right:
                y_top = rect_top
                y_bottom = rect_bottom
                if start_y < end_y:
                    y_top = min(y_top, end_y)
                    y_bottom = max(y_bottom, start_y)
                else:
                    y_top = min(y_top, start_y)
                    y_bottom = max(y_bottom, end_y)

                if y_bottom <= y_top:
                    new_intersection_point = np.array([start_x, y_bottom])
                    if not self.check_point_in_rect_corner(new_intersection_point, rect) and \
                            line.check_point_between_line(new_intersection_point):
                        new_intersection = Intersection(new_intersection_point, rect)
                        intersections.append(new_intersection)

                    new_intersection_point = np.array([start_x, y_top])
                    if not self.check_point_in_rect_corner(new_intersection_point, rect) and \
                            line.check_point_between_line(new_intersection_point):
                        new_intersection = Intersection(new_intersection_point, rect)
                        intersections.append(new_intersection)
        else:
            slope = (end_y - start_y) / (end_x - start_x)
            intercept = start_y - slope * start_x

            x_left = rect_left
            x_right = rect_right
            y_left = intercept + slope * x_left
            y_right = intercept + slope * x_right

            if rect_bottom <= y_left <= rect_top:
                new_intersection_point = np.array([x_left, y_left])
                if not self.check_point_in_rect_corner(new_intersection_point, rect) and \
                        line.check_point_between_line(new_intersection_point):
                    new_intersection = Intersection(new_intersection_point, rect)
                    intersections.append(new_intersection)

            if rect_bottom <= y_right <= rect_top:
                new_intersection_point = np.array([x_right, y_right])
                if not self.check_point_in_rect_corner(new_intersection_point, rect) and \
                        line.check_point_between_line(new_intersection_point):
                    # print("new", new_intersection)
                    # print(rect.corners)
                    # print(self.check_point_in_rect_corner(new_intersection, rect))
                    # print("[x_right, y_right]")
                    new_intersection = Intersection(new_intersection_point, rect)
                    intersections.append(new_intersection)

            y_top = rect_top
            y_bottom = rect_bottom
            x_top = (y_top - intercept) / slope
            x_bottom = (y_bottom - intercept) / slope

            if rect_left <= x_top <= rect_right:
                new_intersection_point = np.array([x_top, y_top])
                if not self.check_point_in_rect_corner(new_intersection_point, rect) and \
                        line.check_point_between_line(new_intersection_point):
                    new_intersection = Intersection(new_intersection_point, rect)
                    intersections.append(new_intersection)
            if rect_left <= x_bottom <= rect_right:
                new_intersection_point = np.array([x_bottom, y_bottom])
                if not self.check_point_in_rect_corner(new_intersection_point, rect) and \
                        line.check_point_between_line(new_intersection_point):
                    new_intersection = Intersection(new_intersection_point, rect)
                    intersections.append(new_intersection)

        if intersections:
            return True, intersections
        else:
            return False, intersections

    def check_line_all_obstacles_intersection(self, line):
        for rect_i in self.inflated_rects:
            intersection, intersections = self.line_rectangle_intersection(line, rect_i)
            inside = rect_i.check_point_inside(line.start) or rect_i.check_point_inside(line.end)
            # if (intersection and not inside) or len(intersections) > 1:
            #     print("rect center", rect_i.center)
            #     return True
        return True

    def nearest_intersection(self):
        all_intersections = []
        line = Line(self.current_start_point, self.goal_point)

        for rect_i in self.inflated_rects:
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
        min_intersection_to_start = self.current_start_point

        for intersection_i in all_intersections:
            distance_to_start = self.distance(intersection_i.point, self.current_start_point)
            if distance_to_start < min_distance_to_start:
                min_distance_to_start = distance_to_start
                min_intersection_to_start = intersection_i
        self.min_intersection = min_intersection_to_start
        # print(self.min_intersection)

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

    def step_toward_corner(self):
        nearest_corner = self.nearest_rect_corner
        distance_to_corner = self.distance(self.current_start_point, nearest_corner)
        step_num = int(distance_to_corner / self.step_size)
        if distance_to_corner < 1e-5:
            distance_to_corner = 1e-5
        vector_to_corner = (nearest_corner - self.current_start_point) / distance_to_corner
        for step_i in range(step_num):
            self.current_start_point = self.current_start_point + vector_to_corner * self.step_size
            self.path.append(self.current_start_point)
        self.current_start_point = nearest_corner
        self.path.append(nearest_corner)

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

    def find_intersection_nearest_corner(self):
        nearest_obstacle = self.min_obstacle
        nearest_intersection_point = self.min_intersection.point

        distance_from_start_to_corner = np.inf
        cost_to_go = np.inf
        nearest_rect_corner = None
        # print("\n=============================")
        for corner in nearest_obstacle.corners:
            # print("current_point", self.current_start_point)
            # print("corner_point", corner.corner)
            # print("corner_visited", corner.visited)
            if (abs(nearest_intersection_point[0] - corner.corner[0]) < 1e-3 or
                    abs(nearest_intersection_point[1] - corner.corner[1]) < 1e-3):
                if self.distance(nearest_intersection_point, corner.corner) + \
                        self.distance(corner.corner, self.goal_point) < cost_to_go:
                    nearest_rect_corner = copy.copy(corner)

                    distance_from_start_to_corner = self.distance(self.current_start_point, corner.corner)
                    cost_to_go = self.distance(nearest_intersection_point, corner.corner) + \
                                 self.distance(corner.corner, self.goal_point)
                    # print("corner", corner.corner)
                    # print("current_point", self.current_start_point)
                    # print("distance_from_start_to_corner", distance_from_start_to_corner)
        self.nearest_rect_corner = nearest_rect_corner.corner
        self.distance_from_start_to_corner = distance_from_start_to_corner
        # print("nearest_corner", self.nearest_rect_corner)

    def find_nearest_corner(self):
        nearest_obstacle = self.min_obstacle

        distance_from_start_to_corner = np.inf
        cost_to_go = np.inf
        nearest_rect_corner = None
        # print("\n=============================")
        for corner in nearest_obstacle.corners:
            # print("current_point", self.current_start_point)
            # print("corner_point", corner.corner)
            # print("corner_visited", corner.visited)
            if not corner.visited and (abs(self.current_start_point[0] - corner.corner[0]) < 1e-3 or
                                       abs(self.current_start_point[1] - corner.corner[1]) < 1e-3):
                if self.distance(self.current_start_point, corner.corner) + \
                        self.distance(corner.corner, self.goal_point) < cost_to_go:
                    nearest_rect_corner = copy.copy(corner)

                    distance_from_start_to_corner = self.distance(self.current_start_point, corner.corner)
                    cost_to_go = self.distance(self.current_start_point, corner.corner) + \
                                 self.distance(corner.corner, self.goal_point)
                    # print("corner", corner.corner)
                    # print("current_point", self.current_start_point)
                    # print("distance_from_start_to_corner", distance_from_start_to_corner)
        for corner in nearest_obstacle.corners:
            if self.distance(nearest_rect_corner.corner, corner.corner) < 1e-3:
                corner.visited = True
        self.nearest_rect_corner = nearest_rect_corner.corner
        self.distance_from_start_to_corner = distance_from_start_to_corner
        # print("nearest_corner", self.nearest_rect_corner)

    def one_step_along_rect(self):
        nearest_rect_corner = self.nearest_rect_corner
        distance_from_start_to_corner = self.distance(self.current_start_point, nearest_rect_corner)
        # print(nearest_rect_corner)
        # print("distance_from_start_to_corner", self.distance_from_start_to_corner)
        if distance_from_start_to_corner < self.step_size:
            self.current_start_point = nearest_rect_corner
            self.path.append(nearest_rect_corner)
        else:
            vector_to_conor = (nearest_rect_corner - self.current_start_point) / distance_from_start_to_corner
            self.current_start_point = self.current_start_point + vector_to_conor * self.step_size
            self.path.append(self.current_start_point)
        # print("current_start_point====", self.current_start_point)

    def run(self):
        while self.distance(self.current_start_point, self.goal_point) > self.step_size:
            # print(self.path[-1])
            self.nearest_intersection()
            if self.min_intersection is None:
                self.step_toward_goal()
            else:
                # print("min_intersection", self.min_intersection.point)
                self.nearest_obstacle()
                # print("min_obstacle_center", self.min_obstacle.center)
                self.find_intersection_nearest_corner()
                line = Line(self.current_start_point, self.nearest_rect_corner)
                if self.check_line_all_obstacles_intersection(line):
                    self.step_toward_intersection()
                    # print("============")
                else:
                    self.step_toward_corner()

                # self.step_toward_intersection()

                line = Line(self.current_start_point, self.goal_point)
                intersection, _ = self.line_rectangle_intersection(line, self.min_obstacle)
                # print("intersections", _[0].point)
                # print(_)
                while intersection:
                    self.find_nearest_corner()
                    # print(self.nearest_rect_corner)
                    while self.distance(self.current_start_point, self.nearest_rect_corner) > self.step_size:
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
                    intersection, _ = self.line_rectangle_intersection(line, self.min_obstacle)
        if self.distance(self.current_start_point, self.goal_point) <= self.step_size:
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

        # final_path = self.path
        # if len(final_path) > 2:
        #     new_path = [final_path[0]]
        #     for i in range(len(final_path) - 1):
        #         current_point = final_path[i]
        #         count = 0
        #         for j in range(i + 1, len(final_path)):
        #             next_point = final_path[j]
        #             line = Line(current_point, next_point)
        #             if self.check_line_all_obstacles_intersection(line):
        #                 print("line", line.start, line.end)
        #                 print("===========collision===========")
        #                 new_path.append(next_point)
        #                 break
        #             count += 1
        #         i += count
        #     new_path.append(final_path[-1])
        # self.path = new_path

    def plot_rectangulars(self, ax):

        for rect_i in self.inflated_rects[0: len(self.obstacle_list)]:
            inflated_rect_i = rect_i
            origin_rect_i = Rectangular(rect_i.center, rect_i.width - 2 * self.inflated_size,
                                        rect_i.height - 2 * self.inflated_size)
            ax = inflated_rect_i.plot_rectangular(ax, color='grey')
            ax = origin_rect_i.plot_rectangular(ax, color='b')

    def plot_path(self, ax):
        path_x = []
        path_y = []
        ax.plot(self.path[0][0], self.path[0][1], color='r', marker='*')
        ax.plot(self.path[-1][0], self.path[-1][1], color='orange', marker='o')
        for path_i in self.path:
            # print(path_i)
            path_x.append(path_i[0])
            path_y.append(path_i[1])
        ax.plot(path_x, path_y, '-g', linewidth=1.5)


def obstacle_adapter(obstacle_list):
    obstacle_list = [
        [
            [
                ob[0] + ob[-1] / 2,
                ob[1] + ob[-1] / 2
            ],
            ob[-1],
            ob[-1]
        ]
        for ob in obstacle_list]
    return obstacle_list


if __name__ == '__main__':
    # obscacles
    # [center_x, center_y], width, height
    obstacle_list = [[199.9946259576258, 83.31695255290691, 54.232595826027215],
                     [8.774308555363069, 218.29587852506296, 54.232595826027215],
                     [98.31832193935307, 45.153090684989294, 54.232595826027215],
                     [101.87145895025849, 179.4797170664781, 54.232595826027215],
                     [217.71835981488442, 183.53196072220774, 54.232595826027215],
                     [12.848261121594538, 38.077922709815525, 54.232595826027215]]

    agent_start = [[250.0, 80.0]]

    agent_end = [[250.0, 300.0]]

    obstacle_list = obstacle_adapter(obstacle_list)

    step_size = 100.0
    inflated_size = 10

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([0, 300])
    ax.set_ylim([0, 300])
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
        print(final_path)
        # print(total_time)
    bug_planner = BugPlanner(agent_start[0], agent_end[0], step_size, inflated_size, obstacle_list)
    bug_planner.plot_rectangulars(ax)

    plt.show()
