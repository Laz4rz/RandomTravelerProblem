import unittest
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class Position:
    def __init__(self, x: int, y: int):
        self.__x = x
        self.__y = y

    """ properties of x """

    @property
    def x(self) -> int:
        return self.__x

    @x.setter
    def x(self, value: int):
        self.__x = value

    """ properties of y """

    @property
    def y(self) -> int:
        return self.__y

    @y.setter
    def y(self, value: int):
        self.__y = value

    """ changing behavior of special attributes and mathematical operations to vector-like """

    def __repr__(self):
        return f"x = {self.__x}, y = {self.__y}"

    def __add__(self, other):
        return Position(self.__x + other.__x, self.__y + other.__y)

    def __sub__(self, other):
        return Position(self.__x - other.__x, self.__y - other.__y)

    def __eq__(self, other):
        return self.__x == other.__x and self.__y == other.__y


class TravelerProblem:
    def __init__(
        self,
        matrix_size: tuple = (100, 100),
        objective: tuple = (50, 50),
        number_of_iterations: int = 10,
    ):
        self.__matrix_size = Position(matrix_size[0], matrix_size[1])
        self.__objective = Position(objective[0], objective[1])
        self.__number_of_iterations = number_of_iterations

    """ properties of matrix_size """

    @property
    def matrix_size(self):
        return self.__matrix_size

    @matrix_size.setter
    def matrix_size(self, size: tuple):
        self.__matrix_size.x = size[0]
        self.__matrix_size.y = size[1]

    """ properties of objective """

    @property
    def objective(self):
        return self.__objective

    @objective.setter
    def objective(self, point: tuple):
        self.__objective.x = point[0]
        self.__objective.y = point[1]

    """ properties of number_of_iterations """

    @property
    def number_of_iterations(self) -> int:
        return self.__number_of_iterations

    @number_of_iterations.setter
    def number_of_iterations(self, value: int):
        self.__number_of_iterations = value

    """ carrying out the experiment """
    """ the Point class objects are converted to numpy arrays for better computational efficiency"""
    carry_exp_desc = (
        f"Returns a list of the number of moves it took to get to the objective in each iteration of "
        f"experiment and the norm of objective - start_position which can help to evaluate the shortest path"
    )

    def carry_experiment(self) -> carry_exp_desc:
        rng = np.random.default_rng()
        start_position = np.array(
            [
                rng.integers(0, self.__matrix_size.x),
                rng.integers(0, self.__matrix_size.y),
            ]
        )
        current_position = np.copy(start_position)
        objective = np.array([self.__objective.x, self.__objective.y])
        shortest_path = objective - start_position
        lower_boundary = np.array([0, 0])
        upper_boundary = np.array([self.__matrix_size.x, self.__matrix_size.y])
        movement = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        number_of_moves = 0
        number_of_moves_list = []

        for _ in tqdm(range(self.__number_of_iterations)):
            while (current_position != objective).any():
                move_choice = rng.integers(0, 4)
                number_of_moves += 1
                if (
                    movement[move_choice] + current_position > upper_boundary
                ).any() or (
                    movement[move_choice] + current_position < lower_boundary
                ).any():
                    continue
                else:
                    current_position += movement[move_choice]
            number_of_moves_list.append(number_of_moves)
            number_of_moves = 0
            current_position = np.copy(start_position)

        """ dynamically setting the attributes and their properties"""
        TravelerProblem.number_of_moves = property(lambda self: number_of_moves_list)
        TravelerProblem.shortest_path = property(lambda self: shortest_path)
        return number_of_moves_list, shortest_path

    def report(
        self,
        number_of_moves_list: list,
        shortest_path: np.ndarray,
        plot_title: str = "The experiment summary",
    ) -> (float, float):
        avg = np.average(number_of_moves_list)
        std = np.std(number_of_moves_list)
        length = len(number_of_moves_list)
        print(
            f"The shortest path to the objective: {shortest_path}, average number of moves: {avg}"
            f", standard deviation: {std}"
        )
        x_axis = np.arange(0, length, 1)
        plt.scatter(x_axis, number_of_moves_list, color="orange")
        plt.xlim(0, length)
        plt.ylim(0, max(number_of_moves_list))
        plt.title(plot_title)
        plt.xlabel("Iteration")
        plt.ylabel("Number or moves")
        plt.axhline(avg, linestyle="dashed", color="red")
        plt.legend(["Average", "Number of moves"])
        plt.tight_layout()
        for i in range(3):
            plt.axhspan(avg + std * i, avg + std + std * i, alpha=0.2 * (i + 1))
            plt.axhspan(avg - std * i, avg - std - std * i, alpha=0.2 * (i + 1))
        plt.show()

        return avg, std


def test():
    try:
        matrix_size = (100, 100)
        objective_coords = (1, 1)
        number_of_iterations = 10
        experiment = TravelerProblem(
            matrix_size, objective_coords, number_of_iterations
        )
        number_of_moves_list, shortest_path = experiment.carry_experiment()
        experiment.report(number_of_moves_list, shortest_path)
        print(bcolors.OKGREEN + "\u2713" + " Test passed" + bcolors.ENDC)
    except ValueError:
        print(bcolors.FAIL + "\u2717" + " Test failed: ValueError" + bcolors.ENDC)
    except Exception:
        print(bcolors.FAIL + "\u2717" + " Test failed: Exception" + bcolors.ENDC)
