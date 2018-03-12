#ifndef CDHMM_H
#define CDHMM_H

#include <set>
#include <unordered_map>

#include "GMM.h"

class Continuous_Density_Hidden_Markov_Model{
private:
	set<int> *state_connection = nullptr;

	string type_covariance;
	string type_model;

	int dimension_event;
	int number_gaussian_components;

	void Search_State_Sequence(int event_index, int state_index, int state_sequence[], vector<int*> gamma);

	bool Access_State(int previous_state_index, int state_index);

	double Backward_Algorithm(int length_event, vector<int> state, double **beta, double likelihood[]);
	double Forward_Algorithm(int length_event, vector<int> state, double **alpha, double likelihood[]);
public:
	vector<string> state_label;

	int number_states;

	double *initial_probability = nullptr;

	unordered_map<int, double> *transition_probability = nullptr;
	unordered_map<int, double> *valid_transition = nullptr;

	Gaussian_Mixture_Model **GMM = nullptr;

	Continuous_Density_Hidden_Markov_Model(string path);
	Continuous_Density_Hidden_Markov_Model(set<int> state_connection[], string type_model, vector<string> state_label, int dimension_event, int number_states, string type_covariance = "", int number_gaussian_components = 0);
	~Continuous_Density_Hidden_Markov_Model();

	void Initialize(int number_events, int length_event[], double **_event);
	void Save_Model(string path);

	double Baum_Welch_Algorithm(int number_events, int length_event[], vector<int> state[], double minimum_variance, double **_event);
	double Evaluation(int length_event, double **_event);
	double Viterbi_Algorithm(string *optimal_label_sequence, int **optimal_state_sequence, int length_event, double *_event);

#ifdef Neural_Networks_H
	double Baum_Welch_Algorithm(int number_events, int length_event[], vector<int> state[], double minimum_variance, double **_event, Neural_Networks *NN);
	double Viterbi_Algorithm(string *optimal_label_sequence, int **optimal_state_sequence, int length_event, double *_event, Neural_Networks *NN);
#endif
};

#endif