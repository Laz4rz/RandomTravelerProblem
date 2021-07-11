from travelerproblem import TravelerProblem


if __name__ == "__main__":
    matrix_size = (100, 100)
    objective_coords = (1, 1)
    number_of_iterations = 1000
    experiment = TravelerProblem(matrix_size, objective_coords, number_of_iterations)
    number_of_moves_list, shortest_path = experiment.carry_experiment()
    experiment.report(number_of_moves_list, shortest_path)
