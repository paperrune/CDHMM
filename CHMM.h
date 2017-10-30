#include "GMM.h"

class Continuous_Hidden_Markov_Model{
private:
	bool **state_connection;

	char type_covariance[16];
	char type_model[8];
	char **state_label;

	int dimension_event;
	int number_gaussian_components;
	int number_states;

	void Search_State_Sequence(int event_index, int state_index, int state_sequence[], int **gamma);

	bool Access_State(int previous_state_index, int state_index);
	
	double Backward_Algorithm(int length_event, int number_states, int state[], double scale[], double **beta, double **likelihood);
	double Forward_Algorithm(int length_event, int number_states, int state[], double scale[], double **alpha, double **likelihood);
public:
	int maximum_length_label;

	double *initial_probability;
	
	double **transition_probability;

	Gaussian_Mixture_Model **GMM;

	Continuous_Hidden_Markov_Model(bool **state_connection, char type_covariance[], char type_model[], char **state_label, int dimension_event, int number_gaussian_components, int number_states);
	~Continuous_Hidden_Markov_Model();

	void Initialize(int number_events, int length_event[], double ***_event, int seed);
	void Load_Parameter(char path[]);
	void Save_Parameter(char path[]);

	double Baum_Welch_Algorithm(int number_events, int length_event[], int number_states[], int **state, double minimum_variance, double ***_event);
	double Evaluation(int length_event, double **_event);
	double Viterbi_Algorithm(char optimal_label_sequence[], int optimal_state_sequence[], int length_event, double **_event);
};
