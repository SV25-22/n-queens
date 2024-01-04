import random
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.widgets import Slider
import numpy as np
from matplotlib.patches import Rectangle

#makes a random board for initialization
def random_board(board_size):
    board = list(range(board_size))
    random.shuffle(board)
    return board

def initialization(population_size,board_size):
    initial_population = []
    for i in range(population_size):
        initial_population.append(random_board(board_size))
    return initial_population

#calculates the fitness of a board
def board_fitness(board):
    size = len(board) 
    safe_pairs = 0
    for i in range(size):
        collumn_i = board[i]
        for j in range(i + 1, size):
            collumn_j = board[j]
            if collumn_i != collumn_j and abs(collumn_i - collumn_j) != abs(i - j):
                safe_pairs += 1
    total_pairs = (size * (size - 1)) // 2
    return safe_pairs / total_pairs

def binary_crossover(m, f):
    size = len(m)
    child_1 = []
    child_2 = []
    for i in range(size):
        if random.choice([True, False]):
            child_1.append(m[i])
            child_2.append(f[i])
        else:
            child_1.append(f[i])
            child_2.append(m[i])
    return child_1,child_2

def single_point_crossover(m,f):
    size = len(m)
    child_1 = []
    child_2 = []
    j = random.randint(1,size-1)
    child_1 = m[:j] + f[j:]
    child_2 = f[:j] + m[j:]
    return child_1,child_2

def mutation(board):
    size = len(board)
    random_queen = random.randint(0, size-1)
    random_collumn = random.randint(0, size-1)
    board[random_queen] = random_collumn
    return board

def roulette_selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    probabilities = []
    for fitness in fitness_values:
        probability = fitness / total_fitness
        probabilities.append(probability)    
    m = random.choices(range(len(population)), weights=probabilities)[0]
    pop2 = population[:m] + population[m+1:]
    prob2 = probabilities[:m] + probabilities[m+1:]
    f = random.choices(range(len(pop2)), weights=prob2)[0]

    return m,f
    #selected_index = random.choices(range(len(population)), weights=probabilities)[0] 
    #return population[selected_index]


def genetic_algorithm(population,population_size,generation_size,mutation_rate,board_size,elitism = True):
    gen_i = 1
    best_boards = []
    statistics = []
    for i in range(generation_size):

        if gen_i % 10 == 0:
            print("Current generation: ",end="")
            print(gen_i)

        fitness_values = []
        for board in population:
            fitness_values.append(board_fitness(board))
        current_best_board_index = fitness_values.index(max(fitness_values))
        current_best_board = population[current_best_board_index]
        current_best_board_fitness = fitness_values[current_best_board_index]


        if(current_best_board_fitness == 1.0):

            print("Solution find in generation: ",end="")
            print(gen_i)
            print(population_size)
            print(len(population))
            value_counts = Counter(fitness_values)
            values = list(value_counts.keys())
            counts = list(value_counts.values())
            statistics.append([values,counts])
            best_boards.append(current_best_board)
            return best_boards,statistics
        

        gen_i+=1
        next_generation = []
        k = 0
        if elitism:
            next_generation.append(current_best_board)
            m,f = roulette_selection(population,fitness_values)
            male = population[m]
            female = population[f]
            #male = roulette_selection(population,fitness_values)
            #female = roulette_selection(population,fitness_values)
            child_1,child_2 = binary_crossover(male,female)
            if random.random() < mutation_rate:
                child_1 = mutation(child_1)
            next_generation.append(child_1)
            k = 1


        for i in range(population_size//2-k):
            m,f = roulette_selection(population,fitness_values)
            male = population[m]
            female = population[f]
            #male = roulette_selection(population,fitness_values)
            #female = roulette_selection(population,fitness_values)
            child_1,child_2 = binary_crossover(male,female)
            if random.random() < mutation_rate:
                child_1 = mutation(child_1)
            if random.random() < mutation_rate:
                child_2 = mutation(child_2)
            next_generation.append(child_1)
            next_generation.append(child_2)
        


        population = next_generation
        value_counts = Counter(fitness_values)
        values = list(value_counts.keys())
        counts = list(value_counts.values())
        statistics.append([values,counts])
        best_boards.append(current_best_board)
    print("Could not find the solution.")
    print(population_size)
    print(len(population))
    return best_boards,statistics

#best_boards[-1] = best_board overall
#len(best_boards) = needed generations

def plot_queens_solution(ax, board):
    n = len(board)
    ax.imshow([[0, 1] * (n // 2), [1, 0] * (n // 2)] * (n // 2), cmap='gray', origin='lower')

    for row, col in enumerate(board):
        ax.text(col, row, '♛', ha='center', va='center', fontsize=20, color='red')

    ax.set_xticks([])
    ax.set_yticks([])


def update(val):
    global current_index
    current_index = int(val) - 1
    x_data, y_data = data_list[current_index]
    scatter.set_offsets(np.column_stack((x_data, y_data)))
    scatter.set_label(f'Generation {current_index + 1}')
    ax.legend()
    fig.canvas.draw_idle()


def is_safe(queens_positions, new_row, new_col):
    for row, col in enumerate(queens_positions):
        if ( new_col == col or abs(new_row - row) == abs(new_col - col) ) and row!=new_row:
            return False
    return True

def draw_board(ax, queens_positions):
    board_size = len(queens_positions)
    ax.clear()
    
    for row in range(board_size):
        for col in range(board_size):
            color = 'white' if (row + col) % 2 == 0 else 'black'
            ax.add_patch(Rectangle((col, row), 1, 1, fill=True, color=color))

    for row, col in enumerate(queens_positions):
        if is_safe(queens_positions, row, col):
            ax.add_patch(Rectangle((col, row), 1, 1, fill=True, color='green'))
        else:
            ax.add_patch(Rectangle((col, row), 1, 1, fill=True, color='red'))

    ax.set_xlim(0, board_size)
    ax.set_ylim(0, board_size)
    ax.set_aspect('equal', adjustable='box')

def update_solution(val, queens_solutions, ax):
    index = int(val)
    queens_positions = queens_solutions[index]
    draw_board(ax, queens_positions)
    plt.draw()

if __name__ == "__main__":
    board_size = 11
    population_size = 3000
    elitism = False
    generation_size = 400
    rez = []
    for i in range(4):
        population = initialization(population_size,board_size)
        best_boards,statistics = genetic_algorithm(population,population_size,generation_size,0.05,board_size,elitism)
        rez.append(len(best_boards))
    print(rez)
    print(rez/200)
    print(best_boards[-1])
    print(statistics[-1])
    data_list = statistics

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    current_index = 0
    x_data, y_data = data_list[current_index]

    scatter = ax.scatter(x_data, y_data, label=f'Generation {current_index + 1}', color='blue')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, population_size)

    slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(slider_ax, 'Select generation', 1, len(data_list), valinit=1, valstep=1)


    
    slider.on_changed(update)

    plt.show()



    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

# Replace this list with your actual list of N-Queens solutions

# Create a slider to choose the solution
    ax_slider = plt.axes([0.1, 0.01, 0.65, 0.03])
    slider = Slider(ax_slider, 'Solution', 0, len(best_boards) - 1, valinit=0, valstep=1)

# Attach the update_solution function to the slider's event
    slider.on_changed(lambda val: update_solution(val, best_boards, ax))

# Initial display
    draw_board(ax, best_boards[0])
    print(best_boards)
    plt.show()

