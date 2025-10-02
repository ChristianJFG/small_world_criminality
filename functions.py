# import libraries 
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
from multiprocessing import Pool

def dispertion_relation(params, plot=False, eigenvalues: list=None, index_critical_eigenvalue: int=None):
    """
    Function to get the dispersion relation of the graph
    """
    # Parameters
    alpha = params['alpha']
    beta = params['beta']
    delta = params['delta']
    do = params['do']
    I = params['I']
    rho_ss = params['rho_ss']
    O_ss = params['O_ss']
    lambda_c = params['lambda_c']
    
        # Define a suitable range of lambda values
    lambda_vals = np.linspace(-8, 0, 500)

    # b(λ)
    def b(lambda_N):
        return ((alpha + beta)/delta
                + (delta * beta)/(alpha + beta)
                - lambda_N * (2 + do))

    # c(λ)
    def c(lambda_N):
        return (
            2 * do * (lambda_N**2)
            + lambda_N * (
                (2 * I * alpha * delta) / (alpha + beta)
                - (2 * delta * beta) / (alpha + beta)
                - (do * (alpha + beta)) / delta
            )
            + (alpha + beta)
        )

    # η(λ) = -b(λ)/2 + (1/2)*sqrt(b(λ)^2 - 4 c(λ))
    def eta(lambda_N):
        return -0.5 * b(lambda_N) + 0.5 * np.sqrt(b(lambda_N)**2 - 4 * c(lambda_N))

    # Evaluate eta on the grid of lambda values
    eta_vals = eta(lambda_vals)

    # Extract only the real part
    eta_vals_real = np.real(eta_vals)

    # Determine the range for positive values
    positive_range = np.where(eta_vals_real > 0)[0]

    min_lambda = lambda_vals[positive_range[0]]
    max_lambda = lambda_vals[positive_range[-1]]

    if plot: 
        # Plot the real part of eta vs. lambda
        plt.figure()
        plt.plot(lambda_vals, eta_vals_real, label=r'Real[$\eta(\lambda)$]')
        # Graph a horizontal line at 0
        plt.axhline(0, color='black', linestyle='-')
        
        # Plot eigenvalues if provided
        if eigenvalues is not None:
            # Plot eigenvalues as asterisks on x-axis
            plt.plot(eigenvalues, [0] * len(eigenvalues), '*', color='red', 
                    markersize=10, label='Eigenvalores')
            
            # If index of critical eigenvalue is provided, highlight it
            if index_critical_eigenvalue is not None:
                plt.plot(eigenvalues[index_critical_eigenvalue], 0, '*', 
                        color='green', markersize=15, 
                        label='Eigenvalor crítico')
        
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'Real[$\eta(\lambda)$]')
        plt.title('Relación de dispersión')
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        'min_lambda': float(min_lambda),
        'max_lambda': float(max_lambda),
    }


def get_dynamics(graph, params, plot=False, hard_integration=False):
    """
    Function to get the dynamics of the graph
    """
    # Parameters
    alpha = params['alpha']
    beta = params['beta']
    delta = params['delta']
    do = params['do']
    I = params['I']
    rho_ss = params['rho_ss']
    O_ss = params['O_ss']
    lambda_c = params['lambda_c']
    G = graph
    n = G.number_of_nodes()
    e = G.number_of_edges()
    A = nx.adjacency_matrix(G).todense()

    
    def system(t, y, A, alpha, beta, delta, do, I):
        """
        y[k][0] is rho
        y[k][1] is O
        n repetitions for each municipality
        2 variables, rho and O
        """


        y = y.reshape(n, 2)
        
        dydt = np.zeros_like(y)

        

        def function_1(j):
            return A[i][j] * (y[j][0] - y[i][0])
        
        def function_2(j):
            return A[i][j] * ((y[i][1]/y[j][1])**I * y[j][0] - (y[j][1]/y[i][1])**I * y[i][0])
        
        def function_3(j):
            return A[i][j] * (y[j][1] - y[i][1])
        
        j = np.arange(16)
        

        for i in range(n):
            f_1_j = np.vectorize(function_1)(j)
            f_2_j = np.vectorize(function_2)(j)
            f_3_j = np.vectorize(function_3)(j)
            dydt[i, 0] = -y[i, 0] * y[i, 1] + alpha + np.sum(f_1_j) + np.sum(f_2_j)
            dydt[i, 1] = -delta * y[i, 1] + beta + y[i, 0] * y[i, 1] + do * np.sum(f_3_j)

        return dydt.flatten()
    # Initial conditions
    np.random.seed(0)
    y0 = np.array([rho_ss, O_ss] * n) + np.random.uniform(-0.01, 0.01, n * 2) * np.array([rho_ss, O_ss] * n)

    # Time span
    t_span = (0, 1000)
    

    # Solve the system of ODEs
    print("Starting the integration...")
    if hard_integration:
        t_eval = np.linspace(*t_span, 10000)
        solution = sp.integrate.solve_ivp(system, t_span, y0, method='RK45', args=(A, alpha, beta, delta, do, I), t_eval=t_eval)
    else:
        t_eval = np.linspace(*t_span, 1000)
        solution = sp.integrate.solve_ivp(system, t_span, y0, method='BDF', args=(A, alpha, beta, delta, do, I), t_eval=t_eval)


    print("Integration finished.")

    t = solution.t
    y = solution.y.reshape(n, 2, -1)

    # Extract the stationary state from the dynamics (the last values)
    nodes = np.arange(1, n + 1)
    last_values_rho = np.zeros(n)
    last_values_O = np.zeros(n)
    for i in range(n):
        last_values_rho[i] = y[i, 0][-1]
        last_values_O[i] = y[i, 1][-1]

    print("Linear algebra computations...")
    # Computing the spectrum of the Laplacian matrix
    L = -1 * nx.laplacian_matrix(G).todense()

    eigenvalues, eigenvectors = np.linalg.eig(L)

    # get the index of the nearest eigenvalue to lambda_c
    index_lambda_c = np.argmin(np.abs(eigenvalues - lambda_c))
    # the index cannot be zero 
    min_lambda, max_lambda = dispertion_relation(params).values()
    # check if the index is in the range
    critical_eigenvalue = eigenvalues[index_lambda_c]
    linear_analysis = True
    
    if critical_eigenvalue < min_lambda or critical_eigenvalue > max_lambda:
        linear_analysis = False
    
    if linear_analysis:
        critical_eigenvector = eigenvectors[:, index_lambda_c]
        
    else: 
        critical_eigenvector = np.zeros(n)
        print("The critical eigenvalue is not in the range of the dispersion relation.")
        print("The linear analysis cannot be performed.")
        print("The critical eigenvalue is: ", critical_eigenvalue)

    dispertion_relation(params, plot=plot, eigenvalues=eigenvalues, index_critical_eigenvalue=index_lambda_c)
    

    if plot:
        
        # Creating the stem plots
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))

        axes[0].stem(nodes, last_values_rho, linefmt='gray', markerfmt='ko', basefmt=' ', bottom=rho_ss)
        axes[0].axhline(rho_ss, color='red', linestyle='--')


        # Labeling the axes
        axes[0].set_xlabel('Nodos')
        axes[0].set_ylabel(r'$\rho$')

        axes[1].stem(nodes, last_values_O, linefmt='gray', markerfmt='ko', basefmt=' ', bottom=O_ss)
        axes[1].axhline(O_ss, color='red', linestyle='--')

        # Labeling the axes
        axes[1].set_xlabel('Nodos')
        axes[1].set_ylabel(r'$O$')

        axes[2].stem(nodes, critical_eigenvector, linefmt='gray', markerfmt='ko', basefmt=' ')
        axes[2].axhline(0, color='red', linestyle='--')

        # Labeling the axes
        axes[2].set_xlabel('Nodos')
        axes[2].set_ylabel('Eigenvector')

        # Display the plot
        plt.show()

    # Variance 
    variance_rho = np.var(last_values_rho)
    variance_O = np.var(last_values_O)
    variance_eigenvector = np.var(critical_eigenvector)

    # Inverse Participation Ratio (IPR)
    ipr_rho = np.sum(last_values_rho**4) / (np.sum(last_values_rho**2)**2)
    ipr_O = np.sum(last_values_O**4) / (np.sum(last_values_O**2)**2)
    ipr_eigenvector = np.sum(critical_eigenvector**4) / (np.sum(critical_eigenvector**2)**2)

    # Network average degree
    k_avg = 2 * e / n

    print("Computations finished.")
    
    



    # Return the results
    return {
        'critical_eigenvalue': float(eigenvalues[index_lambda_c]),
        'last_values_rho': last_values_rho.tolist(),
        'last_values_O': last_values_O.tolist(),
        'critical_eigenvector': critical_eigenvector.tolist(),
        'eigenvalues': eigenvalues.tolist(),
        'variance_rho': float(variance_rho),
        'variance_O': float(variance_O),
        'variance_eigenvector': float(variance_eigenvector),
        'ipr_rho': float(ipr_rho),
        'ipr_O': float(ipr_O),
        'ipr_eigenvector': float(ipr_eigenvector),
        'k_avg': float(k_avg),
    }



def process_graph(args):
    """Helper function to process a single graph with given parameters"""
    p, seed, n_nodes, n_neighbors, params = args
    graph = nx.newman_watts_strogatz_graph(n=n_nodes, k=n_neighbors, p=p, seed=seed)
    max_clustering = 3/4 * (n_neighbors - 2) / (n_neighbors - 1)
    max_length = n_nodes / (2 * n_neighbors)
    
    transitivity = nx.transitivity(graph) / max_clustering
    avg_length = nx.average_shortest_path_length(graph) / max_length
    
    dynamics = get_dynamics(graph, params)

    # Compute the measures for the random erdos renyi graph
    avg_degree = dynamics["k_avg"]
    transitivity_random = (avg_degree / n_nodes) / max_clustering
    avg_length_random = (np.log(n_nodes) / np.log(avg_degree)) / max_length
    sigma = (transitivity / transitivity_random) / (avg_length / avg_length_random)

    # Compute the presence of hubs
    max_degree = max(dict(graph.degree()).values())
    hub_ness = max_degree / avg_degree
    return {
        'transitivity': transitivity,
        'avg_length': avg_length,
        'dynamics': dynamics,
        'sigma': sigma,
        'hub_ness': hub_ness
    }

def experiments(n_nodes, n_neighbors, n_experiments, sample_size, params):
    print("Starting the experiments...")
    transitivity = []
    average_path_length = []
    sigma_data = []
    hub_data = []
    p_values = np.logspace(-3, 0, n_experiments)
    # p_values = np.linspace(0, 1, n_experiments)
    dynamics_data = {}

    # Create process pool
    with Pool() as pool:
        for i, p in enumerate(p_values):
            print(f"Computing graph {i + 1}/{n_experiments}...")
            
            # Prepare arguments for parallel processing
            args = [(p, seed, n_nodes, n_neighbors, params) for seed in range(sample_size)]
            
            # Process graphs in parallel
            results = pool.map(process_graph, args)
            
            # Extract results
            transitivities = [r['transitivity'] for r in results]
            avg_lengths = [r['avg_length'] for r in results]
            dynamics = [r['dynamics'] for r in results]
            sigmas = [r['sigma'] for r in results]
            hubs = [r['hub_ness'] for r in results]
            
            transitivity.append(np.mean(transitivities))
            average_path_length.append(np.mean(avg_lengths))
            dynamics_data[p] = dynamics
            # Save the mean of sigma
            sigma_data.append(np.mean(sigmas))
            # Save the mean of hub_ness
            hub_data.append(np.mean(hubs))


    # Rest of the code remains the same
    data_ipr_rho = [list(map(lambda x: x['ipr_rho'], dynamics_data[p])) for p in p_values]
    data_ipr_O = [list(map(lambda x: x['ipr_O'], dynamics_data[p])) for p in p_values]
    data_ipr_eigenvector = [list(map(lambda x: x['ipr_eigenvector'], dynamics_data[p])) for p in p_values]
    data_variance_rho = [list(map(lambda x: x['variance_rho'], dynamics_data[p])) for p in p_values]
    data_variance_O = [list(map(lambda x: x['variance_O'], dynamics_data[p])) for p in p_values]
    data_variance_eigenvector = [list(map(lambda x: x['variance_eigenvector'], dynamics_data[p])) for p in p_values]
    data_k_avg = [list(map(lambda x: x['k_avg'], dynamics_data[p])) for p in p_values]
    
    data_p = p_values

    # Plot the results
    print("Plotting the results...")
    fig, axes = plt.subplots(5, 1, figsize=(10, 12))

    # Plot transitivity and average path length
    axes[0].plot(p_values, transitivity, label='Coeficiente de agrupamiento')
    axes[0].plot(p_values, average_path_length, label='Distancia media')
    # Plot vertical line for small world condition when True
    axes[0].plot(p_values, sigma_data, label='Small-world-ness')
    axes[0].plot(p_values, hub_data, label='Hub-ness')
    axes[0].set_xlabel('p')
    axes[0].set_xscale('log')
    axes[0].legend()
    axes[0].set_title('Propiedades de la red')
    axes[0].grid(True)

    # Plot dynamics: IPR
    widths = 0.05 * data_p
    # Plot dynamics: IPR (rho and O)
    for i, (x, y, w) in enumerate(zip(data_p, data_ipr_rho, widths)):
        if i == 0:
            axes[1].boxplot(y, positions=[x], widths=[w], boxprops=dict(color='black'), 
                            whiskerprops=dict(color='black'), capprops=dict(color='black'), 
                            medianprops=dict(color='blue'), label=r'$IPR(\rho)$')
        else:
            axes[1].boxplot(y, positions=[x], widths=[w], boxprops=dict(color='black'), 
                            whiskerprops=dict(color='black'), capprops=dict(color='black'), 
                            medianprops=dict(color='blue'))
    for i, (x, y, w) in enumerate(zip(data_p, data_ipr_O, widths)):
        if i == 0:
            axes[1].boxplot(y, positions=[x], widths=[w], boxprops=dict(color='black'), 
                            whiskerprops=dict(color='black'), capprops=dict(color='black'), 
                            medianprops=dict(color='red'), label=r'$IPR(O)$')
        else:
            axes[1].boxplot(y, positions=[x], widths=[w], boxprops=dict(color='black'), 
                            whiskerprops=dict(color='black'), capprops=dict(color='black'), 
                            medianprops=dict(color='red'))
    axes[1].set_xlabel('p')
    axes[1].set_xscale('log')
    axes[1].legend()
    axes[1].set_title(r'$IPR$ calculado de la integración numérica')
    axes[1].grid(True)

    # Plot variance (rho and O) as boxplots
    for i, (x, y, w) in enumerate(zip(data_p, data_variance_rho, widths)):
        if i == 0:
            axes[2].boxplot(y, positions=[x], widths=[w], boxprops=dict(color='black'), 
                            whiskerprops=dict(color='black'), capprops=dict(color='black'), 
                            medianprops=dict(color='blue'), label=r'Varianza $\rho$')
        else:
            axes[2].boxplot(y, positions=[x], widths=[w], boxprops=dict(color='black'), 
                            whiskerprops=dict(color='black'), capprops=dict(color='black'), 
                            medianprops=dict(color='blue'))
    for i, (x, y, w) in enumerate(zip(data_p, data_variance_O, widths)):
        if i == 0:
            axes[2].boxplot(y, positions=[x], widths=[w], boxprops=dict(color='black'), 
                            whiskerprops=dict(color='black'), capprops=dict(color='black'), 
                            medianprops=dict(color='red'), label=r'Varianza $O$')
        else:
            axes[2].boxplot(y, positions=[x], widths=[w], boxprops=dict(color='black'), 
                            whiskerprops=dict(color='black'), capprops=dict(color='black'), 
                            medianprops=dict(color='red'))
    axes[2].set_xlabel('p')
    axes[2].set_xscale('log')
    axes[2].legend()
    axes[2].set_title('Varianza')
    axes[2].grid(True)

    # Plot IPR eigenvector as boxplots
    for i, (x, y, w) in enumerate(zip(data_p, data_ipr_eigenvector, widths)):
        if i == 0:
            axes[3].boxplot(y, positions=[x], widths=[w], label=r'$IPR(\phi_c)$')
        else:
            axes[3].boxplot(y, positions=[x], widths=[w])
    axes[3].set_xlabel('p')
    axes[3].set_xscale('log')
    axes[3].legend()
    axes[3].set_title('IPR del Eigenvector Crítico')
    axes[3].grid(True)

    # Plot variance eigenvector as boxplots
    for i, (x, y, w) in enumerate(zip(data_p, data_variance_eigenvector, widths)):
        if i == 0:
            axes[4].boxplot(y, positions=[x], widths=[w], label=r'Varianza $\phi_c$')
        else:
            axes[4].boxplot(y, positions=[x], widths=[w])
    axes[4].set_xlabel('p')
    axes[4].set_xscale('log')
    axes[4].legend()
    axes[4].set_title('Varianza del Eigenvector Crítico')
    axes[4].grid(True)

    plt.tight_layout()
    plt.show()

    return {
        "dynamics_data": dynamics_data,
        "data_ipr_rho": data_ipr_rho,
        "data_ipr_O": data_ipr_O,
        "data_ipr_eigenvector": data_ipr_eigenvector,
        "data_variance_rho": data_variance_rho,
        "data_variance_O": data_variance_O,
        "data_variance_eigenvector": data_variance_eigenvector,
        "data_k_avg": data_k_avg
    }