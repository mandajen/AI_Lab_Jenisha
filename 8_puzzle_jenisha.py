from collections import deque

def manhattan_distance(state):
    total = 0
    for i in range(3):
        for j in range(3):
            tile = state[i][j]
            if tile != 0:
                target_row = (tile - 1) // 3
                target_col = (tile - 1) % 3
                total += abs(i - target_row) + abs(j - target_col)
    return total

def solve_8_puzzle(initial_state):
    goal_state = ((1, 2, 3), (4, 5, 6), (7, 8, 0))
    initial_tuple = tuple(tuple(row) for row in initial_state)
    
    if initial_tuple == goal_state:
        return []
    
    visited = set()
    queue = deque()
    queue.append((initial_tuple, []))
    visited.add(initial_tuple)
    
    print("Initial state:")
    for row in initial_tuple:
        print(row)
    print(f"Heuristic (Manhattan distance): {manhattan_distance(initial_tuple)}\n")
    
    print("Exploring states:")
    directions = [(-1, 0, 'up'), (1, 0, 'down'), (0, -1, 'left'), (0, 1, 'right')]
    
    while queue:
        state, path = queue.popleft()
        h_val = manhattan_distance(state)
        print("State:")
        for row in state:
            print(row)
        print(f"Heuristic: {h_val}\n")
        
        if state == goal_state:
            return path
        
        i, j = -1, -1
        for r in range(3):
            for c in range(3):
                if state[r][c] == 0:
                    i, j = r, c
                    break
            if i != -1:
                break
        
        for dr, dc, move in directions:
            ni, nj = i + dr, j + dc
            if 0 <= ni < 3 and 0 <= nj < 3:
                state_list = [list(row) for row in state]
                state_list[i][j], state_list[ni][nj] = state_list[ni][nj], state_list[i][j]
                new_state = tuple(tuple(row) for row in state_list)
                
                if new_state not in visited:
                    visited.add(new_state)
                    new_path = path + [move]
                    queue.append((new_state, new_path))
    
    return None

if __name__ == "__main__":
    initial_state = [[1, 2, 3], [4, 0, 5], [7, 8, 6]]
    solution_path = solve_8_puzzle(initial_state)
    
    if solution_path is None:
        print("No solution found.")
    else:
        print("Optimal solution path:")
        print(solution_path)