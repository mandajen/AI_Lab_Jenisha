def heuristic(state, goal):
    return sum(1 for s, g in zip(state, goal) if s != g)

def hill_climbing(initial_state, goal_state):
    current_state = initial_state.copy()
    path = [current_state.copy()]
    h_values = [heuristic(current_state, goal_state)]
    
    while True:
        print(f"Current State: {current_state}, Heuristic: {h_values[-1]}")
        
        if h_values[-1] == 0:
            print("\nGoal Reached!")
            return path, h_values
        
        neighbors = []
        for i in range(len(current_state) - 1):
            new_state = current_state.copy()
            new_state[i], new_state[i+1] = new_state[i+1], new_state[i]  # Swap adjacent
            neighbors.append((new_state, heuristic(new_state, goal_state)))
        
        best_neighbor = min(neighbors, key=lambda x: x[1])
        
        if best_neighbor[1] >= h_values[-1]:
            print("\nStuck in Local Minimum!")
            return path, h_values
        
        current_state = best_neighbor[0]
        path.append(current_state.copy())
        h_values.append(best_neighbor[1])

if __name__ == "__main__":
    initial_state = ['C', 'A', 'D', 'B']
    goal_state = ['A', 'B', 'C', 'D']
    
    print(f"Initial Stack: {initial_state}")
    print(f"Goal State: {goal_state}\n")
    
    path, h_values = hill_climbing(initial_state, goal_state)
    
    print("\nSolution Path:")
    for step, state in enumerate(path):
        print(f"Step {step}: {state} (h={h_values[step]})")
    
    if h_values[-1] == 0:
        print("\nSuccess! Goal state reached.")
    else:
        print("\nFailed to reach the goal state (stuck in local minimum).")