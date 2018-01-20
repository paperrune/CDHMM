#include <unordered_map>

#include "GMM.h"

class Continuous_Hidden_Markov_Model{
private:
	unordered_map<int, bool> *state_connection;

	string type_covariance;
	string type_model;

	int dimension_event;
	int number_gaussian_components;

	void Search_State_Sequence(int event_index, int state_index, int state_sequence[], int **gamma);

	bool Access_State(int previous_state_index, int state_index);

	double Backward_Algorithm(int length_event, vector<int> state, double **beta, double **likelihood);
	double Forward_Algorithm(int length_event, vector<int> state, double **alpha, double **likelihood);

public:
	vector<string> state_label;

	int number_states;

	double *initial_probability;

	unordered_map<int, double> *transition_probability;
	unordered_map<int, double> *valid_transition;

	Gaussian_Mixture_Model **GMM;

	Continuous_Hidden_Markov_Model(string path);
	Continuous_Hidden_Markov_Model(unordered_map<int, bool> state_connection[], string type_covariance, string type_model, vector<string> state_label, int dimension_event, int number_gaussian_components, int number_states);
	~Continuous_Hidden_Markov_Model();

	void Initialize(int number_events, int length_event[], double ***_event);
	void Load_Model(string path);
	void Save_Model(string path);

	double Baum_Welch_Algorithm(int number_events, int length_event[], vector<int> state[], double minimum_variance, double ***_event);
	double Evaluation(int length_event, double **_event);
	double Viterbi_Algorithm(string *optimal_label_sequence, int **optimal_state_sequence, int length_event, double **_event);
};