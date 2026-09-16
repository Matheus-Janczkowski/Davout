# Routine to test the implementation of optimization methods

import numpy as np

from scipy.optimize import curve_fit

from ...PythonicUtilities.path_tools import get_parent_path_of_file

from ...PythonicUtilities.testing_tools import run_class_of_tests

from ...GraphUtilities import plotting_tools

# Defines a function to test the ANN optimization wrappers

class TestPlots:

    def __init__(self):

        # Sets the unimodal data

        self.unimodal_x_data = np.sort(np.random.rand(15))

        self.unimodal_y_data = [np.exp(x) for x in self.unimodal_x_data]

        self.unimodal_y_data2 = [np.exp(1.5*x) for x in self.unimodal_x_data]

        # Sets the multimodal data

        self.multimodal_x_data = []

        self.multimodal_y_data = []

        self.n_curves = 3

        for i in range(self.n_curves):

            self.multimodal_x_data.append(np.sort(np.random.rand(15)))

            self.multimodal_y_data.append([np.exp((i+1)*x) for x in (
            self.multimodal_x_data[-1])])

    def test_multimodal_plot(self):

        print("\n#####################################################"+
        "###################\n#                           Multimodal c"+
        "urve                           #\n###########################"+
        "#############################################\n")

        x_data = [np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])]

        y_data = [np.array([1.0, 2.0, 3.0]), np.array([1.0, 4.0, 6.0])]

        plotting_tools.plane_plot(x_data=x_data, y_data=y_data, 
        file_name="test_two_curves_numpy", plot_type=["line", "scatter"],
        element_size=[1.5, 10.0], color=["yellow", "black"])

        x_data = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]

        y_data = [[1.0, 2.0, 3.0], [1.0, 4.0, 6.0]]

        plotting_tools.plane_plot(x_data=x_data, y_data=y_data, 
        file_name="test_two_curves_list", label=["Exponential $a=24.83$", 
        "Numerical data"])

    # Defines a function to test the plot of a curve with error bar
    
    def test_error_bar(self):

        print("\n#####################################################"+
        "###################\n#                              Error bar"+
        "                               #\n###########################"+
        "#############################################\n")

        # Initializes the error bar with the confidence intervals

        error_bar = [0.1 for i in range(len(self.unimodal_x_data))]

        # Calls the plotter

        # Scatter single curve given the error bar but separately plotted

        print("\nTests error plot with error bar being separately plot"+
        "ted")

        plot_object = plotting_tools.plane_plot(x_data=
        self.unimodal_x_data, y_data=self.unimodal_y_data, file_name=
        "test_error_bar_separate_plotting", plot_type="scatter", 
        error_bar=error_bar, color="black")

        plotting_tools.plane_plot(x_data=self.unimodal_x_data, y_data=
        self.unimodal_y_data2, file_name="test_error_bar_separate_plot"+
        "ting", plot_type="line", plot_object=plot_object)

        # Scatter single curve given the error bar

        plotting_tools.plane_plot(x_data=self.unimodal_x_data, y_data=
        self.unimodal_y_data, error_bar=error_bar, file_name=
        "test_error_bar", plot_type="scatter")

        # Scatter single curve given the error bar with upper and lower
        # bounds for error

        error_bar_lower_upper = [[0.1, 0.05] for i in range(len(
        self.unimodal_x_data))]

        print("\nTests error plot with error bar that has upper and lo"+
        "wer bounds")

        plotting_tools.plane_plot(x_data=self.unimodal_x_data, y_data=
        self.unimodal_y_data, error_bar=error_bar_lower_upper, 
        file_name="test_error_bar_lower_upper", plot_type="scatter")

        # Scatter single curve given the error bar with upper and lower
        # bounds for error

        error_bar_lower_upper = [[0.1, 0.05] for i in range(len(
        self.unimodal_x_data))]

        print("\nTests error plot with error region that has upper and lo"+
        "wer bounds")

        plotting_tools.plane_plot(x_data=self.unimodal_x_data, y_data=
        self.unimodal_y_data, error_bar=error_bar_lower_upper, 
        file_name="test_error_bar_line_lower_upper", plot_type="line")

        # Continuous single curve given the error bar

        print("\nTests error plot for a continuous curve")

        plotting_tools.plane_plot(x_data=self.unimodal_x_data, y_data=
        self.unimodal_y_data, error_bar=error_bar, file_name=
        "test_error_region", plot_type="line")

        # Calls with multimodal data

        error_bar = []

        error_bar_lower_upper = []

        for i in range(self.n_curves):

            error_bar.append([((i+1)/10) for j in range(len(
            self.multimodal_x_data[i]))])

            error_bar_lower_upper.append([[((i+1)/10), ((i+1)/5)] for (
            j) in range(len(self.multimodal_x_data[i]))])

        # Continuous multiple curves given the error bar

        print("\nTests error plot for multiple continuous curve")

        plotting_tools.plane_plot(x_data=self.multimodal_x_data, y_data=
        self.multimodal_y_data, error_bar=error_bar, file_name=
        "test_error_region_multimodal", plot_type="line")

        # Continuous multiple curves given the error bar with upper and
        # lower bounds

        print("\nTests error plot for multiple continuous curve with u"+
        "pper and lower bounds")

        plotting_tools.plane_plot(x_data=self.multimodal_x_data, y_data=
        self.multimodal_y_data, error_bar=error_bar_lower_upper, file_name=
        "test_error_region_multimodal_upper_lower", plot_type="line")

        # Continuous single curve automatically evaluating the error bar
        # for the t-Student distribution

        print("\nTests error plot for a continuous curve and automatic"+
        " evaluation of the error bar following t-Student")

        plotting_tools.plane_plot(x_data=self.unimodal_x_data, y_data=
        self.multimodal_y_data, error_bar="t-Student", file_name="test_e"+
        "rror_region_t_student", plot_type="line")

        # Continuous single curve automatically evaluating the error bar
        # for the normal distribution

        print("\nTests error plot for a continuous curve and automatic"+
        " evaluation of the error bar following normal distribution")

        plotting_tools.plane_plot(x_data=self.unimodal_x_data, y_data=
        self.multimodal_y_data, error_bar="normal distribution", 
        file_name="test_error_region_z_score", plot_type="line")

        # Scatter single curve automatically evaluating the error bar
        # for the normal distribution

        print("\nTests error plot for a scatter curve and automatic ev"+
        "aluation of the error bar following t-Student")

        plotting_tools.plane_plot(x_data=self.unimodal_x_data, y_data=
        self.multimodal_y_data, error_bar="normal distribution", 
        file_name="test_error_bar_z_score", plot_type="scatter")

        # Continuous single curve automatically evaluating the error bar
        # for the t-Student distribution asking for a 90% confidence

        print("\nTests error plot for a scatter curve and automatic ev"+
        "aluation of the error bar following t-Student and 90 percent "+
        "of confidence")

        plotting_tools.plane_plot(x_data=self.unimodal_x_data, y_data=
        self.multimodal_y_data, error_bar={"name": "t-Student", "confi"+
        "dence": 0.9}, file_name="test_error_region_t_student_0_90", 
        plot_type="line")

        # Continuous single curve automatically evaluating the error bar
        # for the normal distribution asking for a 90% confidence

        print("\nTests error plot for a scatter curve and automatic ev"+
        "aluation of the error bar following normal distribution and 9"+
        "0 percent of confidence")

        plotting_tools.plane_plot(x_data=self.unimodal_x_data, y_data=
        self.multimodal_y_data, error_bar={"name": "normal distribution",
        "confidence": 0.9}, file_name="test_error_region_z_score_0_90", 
        plot_type="line")

    def test_matrix_plot(self):

        print("\n#####################################################"+
        "###################\n#                              Matrix pl"+
        "ot                             #\n###########################"+
        "#############################################\n")

        generic_matrix = []

        n = 20

        for i in range(n):

            generic_matrix.append([])

            for j in range(n):

                generic_matrix[-1].append(((i+1)*(j+1))/(n**2))

        plotting_tools.plot_matrix(generic_matrix, 
        get_parent_path_of_file(), "generic matrix")

    def test_mutliple_scattered_ellipses(self):

        print("\n#####################################################"+
        "###################\n#                      Multiple scattere"+
        "d ellipses                     #\n###########################"+
        "#############################################\n")

        limits = [[[-3.0, 1.0], [-1.0, 1.0]], [[-0.5, 1.5], [-1.0, 1.0]]]

        n_ellipses = len(limits)

        n_points = 300

        x_data = []

        y_data = []

        labels = ["E = "+str(j+1) for j in range(n_ellipses)]

        for j in range(n_ellipses):

            x_data.append([])

            y_data.append([])

            for i in range(n_points):

                # Gets a random direction

                random_point = np.random.randn(2)

                # Normalizes it

                random_point = ((1/np.linalg.norm(random_point))*
                random_point)

                # Iterpolates it by the limits

                x_data[-1].append((0.5*(1-random_point[0])*limits[j][0][
                0])+(0.5*(1+random_point[0])*limits[j][0][1]))

                y_data[-1].append((0.5*(1-random_point[1])*limits[j][1][
                0])+(0.5*(1+random_point[1])*limits[j][1][1]))

        plotting_tools.plane_plot("2D_ellipse", x_data=x_data, y_data=
        y_data, plot_type="scatter", element_size=2.5, label=labels, 
        title="$E^{p}\\left(\mathbf{D},\mathbf{x}_{c}\\right)$",
        color_map="coolwarm", aspect_ratio=1.0)

    def test_curve_fitting(self):

        #############################################################
        # Defines a cdde that generates the graph of the rate failures
        # of the simulation, to comprove that increasing the p exponent 
        # of the Lp norm, the parameters set region approaches to the 
        # hypercube topology
        #############################################################

        p = [2, 4, 8, 16, 32]

        # Dados feitos no lagrange

        #error_rate = [0.12, 9.12, 18.58, 22.38, 24.29]

        # Dados feitos no pc do rafael

        error_rate = [0.12, 6.66, 17.82, 22.89, 24.71]

        def Logistic(x, a, b, c):

            return(a/(1+(1/(np.exp(b*x+c)))))

        def Hiperbole(x, a, b,c):

            return (c/(x+b))+a

        def Exponential(x, a, b, c):

            return (a-1/np.exp((b*x)+c))

        curve_function = Exponential

        # 2. Aplicar a regressão usando o curve_fit
        # 'popt' retornará os parâmetros otimizados (a, b)
        # 'pcov' retornará a matriz de covariância (incerteza do ajuste)

        popt, pcov = curve_fit(curve_function, p, error_rate)

        # Extraindo os coeficientes encontrados
        a_opt, b_opt, c_opt = popt

        print(f"Coeficientes encontrados: a = {a_opt:.4f}, b = {b_opt:.4f}, c = {c_opt:.4f}")

        # 3. Gerar a curva ajustada para o plot com uma malha fina
        x_fit = np.linspace(min(p), max(p), 500)
        y_fit = curve_function(x_fit, a_opt, b_opt, c_opt)

        p1=plotting_tools.plane_plot(
                x_data=x_fit,#[np.array(p), x_fit], 
                y_data=y_fit,#[np.array(error_rate), y_fit], 
                file_name="error_rate_failure_Lp_norm.pdf", 
                parent_path=get_parent_path_of_file(),  # Saves in the folder path
                color_map="coolwarm", 
                color=1.0,
                label=str(curve_function.__name__)+" $a="+str(round(a_opt, 2))+"$",
                verbose=True, 
                highlight_points=False,
                x_label="$p$", 
                y_label="Error rate $\\Brackets{\\%}$", 
                x_ticksLabels=p,
                y_ticksLabels=error_rate,
                transparent_background=True,
                latex_package="[nohyperref]{LaTeXUtilities}", 
                dpi=1000
            )

        plotting_tools.plane_plot(
                x_data=p,#[np.array(p), x_fit], 
                y_data=error_rate,#[np.array(error_rate), y_fit], 
                file_name="error_rate_failure_Lp_norm.png", 
                parent_path=get_parent_path_of_file(),  # Saves in the folder path
                color_map="coolwarm", 
                color=0.0,
                element_style="x",
                element_size=5.0,
                plot_type="scatter",
                verbose=True, 
                label="Numerical data",
                x_label="$p$", 
                y_label="Error rate $\\Brackets{\\%}$", 
                x_ticksLabels=p,
                y_ticksLabels=error_rate,
                latex_package="[nohyperref]{LaTeXUtilities}",
                transparent_background=True,
                dpi=1000,
                plot_object=p1
            )

# Runs all tests

if __name__=="__main__":

    class_of_tests = TestPlots()

    run_class_of_tests(class_of_tests)