import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

seed = 21641 #for reproducibility
np.random.seed(seed)

#Initialixing true values and initial values as lists
true_V = [0/8, 1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8, 8/8]
initial_V = [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0] 

def td_method(V, alpha = 0.05):
    state = 4 #episode always starts at the middle state
    while True:
        action = np.random.choice([-1, 1]) #action A - going right or left
        new_state = state + action #observe new state S'
        reward = 1.0 if new_state == 8 else 0.0 #observe reward R
        
        #make the TD(0) update
        V[state] = V[state] + alpha * (reward + V[new_state] -V[state])
        state = new_state #update the state value: S<-S'
        #check if S is terminal
        if state == 0 or state == 8:   
            break

def mc_method(V, alpha = 0.02):
    state = 4 #epsisode always start at the middle state
    state_sequnce = [state]
    returns = 0.0
    while True:
        action = np.random.choice([-1, 1]) #action A - going right or left
        state += action #observe new state S'
        state_sequnce.append(state)
        #check if S is terminal and calculate returns
        if state == 0:
            returns = 0.0
            break
        elif state == 8:
            returns = 1.0       
            break
    #make the Monte Carlo update
    for state in state_sequnce[:-1]:
        V[state] = V[state] + alpha * (returns - V[state])

def calculate_state_values(method = 1):
    #plot the true values
    plt.plot(("1", "2", "3", "4", "5", "6", "7"), true_V[1:8], label='True values') 
    #plot the changing estimates as the number of episodes increses
    n_episodes = [0, 3, 30, 200]
    current_V = initial_V.copy()
    for i in range(201):
        if i in n_episodes:
            plt.plot(("1", "2", "3", "4", "5", "6", "7"),\
                    current_V[1:8], label=str(i) + ' episodes')
        td_method(current_V) if method == 'TD' else mc_method(current_V)
    plt.xlabel('States')
    plt.ylabel('Estimated Values')
    plt.legend()

def calculate_mean_abs_error():
    td_alphas = [0.15, 0.1, 0.05] 
    mc_alphas = [0.02, 0.03, 0.04]
    n_episodes = 201
    runs = 200
    #Calculate td(0) errors for different values of alphas
    for alpha in td_alphas: 
        #array to store state-averaged errors for each episode averaged over runs
        total_errors = np.zeros(n_episodes) 
        for _ in range(runs):
            #array to store errors for each episode averaged over states
            errors = [] 
            current_V = np.copy(initial_V)
            for _ in range(n_episodes):
                #append the error averaged over 7 states
                errors.append(np.sum(np.abs(true_V[1:8] - current_V[1:8]))/ 7.0)
                td_method(current_V, alpha=alpha)
            #sum the errors for every episode when a run is complete
            total_errors += np.asarray(errors) 
        total_errors /= runs #average errors over runs
        plt.plot(total_errors, linestyle='solid',\
            label= 'TD, $\\alpha$ = %.02f' % (alpha))
    for alpha in mc_alphas:
        #array to store state-averaged errors for each episode averaged over runs
        total_errors = np.zeros(n_episodes)
        for _ in range(runs):
            #array to store errors for each episode averaged over states
            errors = [] 
            current_V = np.copy(initial_V)
            for _ in range(n_episodes):
                #append the error averaged over 7 states
                errors.append(np.sum(np.abs(true_V[1:8] - current_V[1:8])) / 7.0)
                mc_method(current_V, alpha=alpha)
            #sum the errors for every episode when a run is complete
            total_errors += np.asarray(errors)
        total_errors /= runs #average errors over runs
        plt.plot(total_errors, linestyle='dashed',\
                label= 'MC, $\\alpha$ = %.02f' % (alpha))
    plt.xlabel('Episodes')
    plt.ylabel('Average Absolute Errors')
    plt.legend()
    
def plot_figures():
    # First plot: MC Estimates
    plt.figure(figsize=(10, 6))
    calculate_state_values(method='MC')  
    plt.title("Monte Carlo Estimates")
    plt.savefig('Monte_Carlo_Estimates.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Second plot: TD(0) Estimates
    plt.figure(figsize=(10, 6)) 
    calculate_state_values(method='TD') 
    plt.title("TD(0) Estimates")
    plt.savefig('TD(0)_Estimates.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Third plot: Comparison of Mean Absolute Errors
    plt.figure(figsize=(10, 6)) 
    calculate_mean_abs_error() 
    plt.title("Comparison of MC vs TD(0) Absolute Errors")
    plt.savefig('MC_vs_TD_abs_errors.png', dpi=300, bbox_inches='tight')
    plt.close()  

if __name__ == '__main__':
    plot_figures()