#include <fstream>
#include <iostream>

#include "CHMM.h"
#include "Kmeans.h"

void Continuous_Hidden_Markov_Model::Search_State_Sequence(int event_index, int state_index, int state_sequence[], vector<int*> gamma){
	state_sequence[event_index] = state_index;
	for (int t = event_index - 1; t >= 0; t--){
		state_sequence[t] = gamma[t + 1][state_sequence[t + 1]];
	}
}

bool Continuous_Hidden_Markov_Model::Access_State(int previous_state_index, int state_index){
	int i = previous_state_index;
	int j = state_index;

	return ((type_model == "ergodic") || (j - i == 0) || (j - i == 1) || (j - i == 2));
}

double Continuous_Hidden_Markov_Model::Backward_Algorithm(int length_event, vector<int> state, double **beta, double **likelihood){
	double log_likelihood = 0;

	for (int t = length_event - 1; t >= 0; t--){
		double scale = 0;

		if (t == length_event - 1){
			for (int i = 0; i < state.size(); i++){
				scale += (beta[t][i] = (type_model == "ergodic" || i == state.size() - 1));
			}
		}
		else
		if (t <= length_event - 2){
			for (int i = 0; i < state.size(); i++){
				double sum = 0;

				for (int j = 0; j < state.size(); j++){
					int k = state[i];
					int l = state[j];

					if (state_connection[k].find(l) != state_connection[k].end() && Access_State(i, j)){
						sum += transition_probability[k].find(l)->second * likelihood[t + 1][j] * beta[t + 1][j];
					}
				}
				scale += (beta[t][i] = sum);
			}
		}
		if (!_finite(log(scale = 1 / scale)) || _isnan(log(scale))){
			cerr << "[Backward Algorithm] [scale: " << scale << "]" << endl;
			return 0;
		}
		for (int i = 0; i < state.size(); i++){
			beta[t][i] *= scale;
		}
		// log_likelihood += log(scale);
	}
	return -log_likelihood;
}
double Continuous_Hidden_Markov_Model::Forward_Algorithm(int length_event, vector<int> state, double **alpha, double **likelihood){
	double log_likelihood = 0;

	for (int t = 0; t < length_event; t++){
		double scale = 0;

		if (t == 0){
			for (int i = 0; i < state.size(); i++){
				int j = state[i];

				scale += (alpha[t][i] = (type_model == "ergodic" || i == 0) * initial_probability[j] * likelihood[t][i]);
			}
		}
		else
		if (t >= 1){
			for (int i = 0; i < state.size(); i++){
				double sum = 0;

				for (int j = 0; j < state.size(); j++){
					int k = state[i];
					int l = state[j];

					if (state_connection[l].find(k) != state_connection[l].end() && Access_State(j, i)){
						sum += alpha[t - 1][j] * transition_probability[l].find(k)->second;
					}
				}
				scale += (alpha[t][i] = sum * likelihood[t][i]);
			}
		}

		// scale
		if (!_finite(log(scale = 1 / scale)) || _isnan(log(scale))){
			cerr << "[Forward Algorithm] [scale: " << scale << "]" << endl;
			return 0;
		}
		for (int i = 0; i < state.size(); i++){
			alpha[t][i] *= scale;
		}
		log_likelihood += log(scale);
	}
	return -log_likelihood;
}

Continuous_Hidden_Markov_Model::Continuous_Hidden_Markov_Model(string path) {
	number_states = 0;
	Load_Model(path);
}
Continuous_Hidden_Markov_Model::Continuous_Hidden_Markov_Model(unordered_map<int, bool> state_connection[], string type_covariance, string type_model, vector<string> state_label, int dimension_event, int number_gaussian_components, int number_states){
	this->type_covariance	= type_covariance;
	this->type_model		= type_model;
	this->dimension_event	= dimension_event;
	this->number_gaussian_components = number_gaussian_components;
	this->number_states		= number_states;
	this->state_connection	= new unordered_map<int, bool>[number_states];

	for (int i = 0; i < number_states; i++){
		this->state_connection[i].insert(state_connection[i].begin(), state_connection[i].end());
		this->state_label.push_back(state_label[i]);
	}

	initial_probability		= new double[number_states];
	transition_probability	= new unordered_map<int, double>[number_states];
	valid_transition		= new unordered_map<int, double>[number_states];

	GMM = new Gaussian_Mixture_Model*[number_states];

	for (int i = 0; i < number_states; i++){
		GMM[i] = new Gaussian_Mixture_Model(type_covariance, dimension_event, number_gaussian_components);
	}
}
Continuous_Hidden_Markov_Model::~Continuous_Hidden_Markov_Model(){
	if (number_states) {
		for (int i = 0; i < number_states; i++) {
			state_connection[i].clear();
		}
		delete[] state_connection;
		state_label.clear();

		for (int i = 0; i < number_states; i++) {
			transition_probability[i].clear();
			valid_transition[i].clear();
			delete GMM[i];
		}
		delete[] initial_probability;
		delete[] transition_probability;
		delete[] valid_transition;
		delete GMM;
	}
}

void Continuous_Hidden_Markov_Model::Initialize(int number_events, int length_event[], double ***_event){
	int number_data = 0;

	double **data;

	KMeans kmeans = KMeans(dimension_event, number_gaussian_components);

	for (int i = 0; i < number_states; i++){
		initial_probability[i] = 1.0 / number_states;
	}

	for (int i = 0; i < number_events; i++){
		number_data += length_event[i];
	}
	data = new double*[number_data];

	for (int i = 0, index = 0; i < number_events; i++){
		for (int j = 0; j < length_event[i]; j++){
			data[index++] = _event[i][j];
		}
	}

	kmeans.Initialize(number_data, data);
	while (kmeans.Cluster(number_data, data));

	for (int i = 0; i < this->number_states; i++){
		for (int j = 0; j < number_gaussian_components; j++){
			for (int k = 0; k < dimension_event; k++){
				if (type_covariance == "diagonal"){
					GMM[i]->diagonal_covariance[j][k] = 1;
				}
				else{
					for (int l = 0; l < dimension_event; l++){
						GMM[i]->covariance[j][k][l] = (k == l);
					}
				}
				GMM[i]->mean[j][k] = kmeans.centroid[j][k];
			}
			GMM[i]->weight[j] = 1.0 / number_gaussian_components;
		}
	}
	delete[] data;
}
void Continuous_Hidden_Markov_Model::Load_Model(string path){
	ifstream file(path);

	if (file.is_open()){
		if (number_states) {
			for (int i = 0; i < number_states; i++) {
				state_connection[i].clear();
			}
			delete[] state_connection;
			state_label.clear();

			for (int i = 0; i < number_states; i++) {
				transition_probability[i].clear();
				valid_transition[i].clear();
				delete GMM[i];
			}
			delete[] initial_probability;
			delete[] transition_probability;
			delete[] valid_transition;
			delete GMM;
		}

		file >> type_covariance;
		file >> type_model;
		file >> dimension_event;
		file >> number_gaussian_components;
		file >> number_states;

		state_connection = new unordered_map<int, bool>[number_states];
		initial_probability = new double[number_states];
		transition_probability = new unordered_map<int, double>[number_states];
		valid_transition = new unordered_map<int, double>[number_states];

		GMM = new Gaussian_Mixture_Model*[number_states];

		for (int i = 0; i < number_states; i++) {
			GMM[i] = new Gaussian_Mixture_Model(type_covariance, dimension_event, number_gaussian_components);
		}

		for (int i = 0, size; i < number_states; i++){
			file >> size;

			for (int j = 0, id; j < size; j++){
				file >> id;
				state_connection[i].insert(pair<int, bool>(id, true));
			}
		}
		for (int i = 0; i < number_states; i++){
			string label;

			if (i == 0) {
				getline(file, label);
			}
			getline(file, label);
			state_label.push_back(label);
		}

		for (int i = 0; i < number_states; i++){
			file >> initial_probability[i];
		}
		for (int i = 0, size; i < number_states; i++){
			file >> size;

			for (int j = 0, id; j < size; j++){
				double value;

				file >> id >> value;
				transition_probability[i].insert(pair<int, double>(id, value));
			}
		}
		for (int i = 0; i < number_states; i++) {
			valid_transition[i].clear();
		}
		for (int i = 0; i < number_states; i++){
			for (auto p = transition_probability[i].begin(); p != transition_probability[i].end(); p++){
				if (p->second){
					valid_transition[p->first].insert(pair<int, double>(i, p->second));
				}
			}
		}

		for (int i = 0; i < number_states; i++){
			for (int j = 0; j < number_gaussian_components; j++){
				file >> GMM[i]->weight[j];
			}
		}

		for (int i = 0; i < number_states; i++){
			for (int j = 0; j < number_gaussian_components; j++){
				for (int k = 0; k < dimension_event; k++){
					file >> GMM[i]->mean[j][k];
				}
			}
		}
		for (int i = 0; i < number_states; i++){
			for (int j = 0; j < number_gaussian_components; j++){
				for (int k = 0; k < dimension_event; k++){
					if (type_covariance == "diagonal"){
						file >> GMM[i]->diagonal_covariance[j][k];
					}
					else{
						for (int l = 0; l < dimension_event; l++){
							file >> GMM[i]->covariance[j][k][l];
						}
					}
				}
			}
		}
		file.close();
	}
	else{
		cerr << "[Load_Model], " + path + " not found" << endl;
	}
}
void Continuous_Hidden_Markov_Model::Save_Model(string path) {
	ofstream file(path);

	file << type_covariance << endl;
	file << type_model << endl;
	file << dimension_event << endl;
	file << number_gaussian_components << endl;
	file << number_states << endl;

	for (int i = 0; i < number_states; i++) {
		file << state_connection[i].size() << endl;

		for (auto c = state_connection[i].begin(); c != state_connection[i].end(); c++) {
			file << c->first << endl;
		}
	}
	for (int i = 0; i < number_states; i++) {
		file << state_label[i] << endl;
	}

	for (int i = 0; i < number_states; i++) {
		file << initial_probability[i] << endl;
	}
	for (int i = 0; i < number_states; i++) {
		file << transition_probability[i].size() << endl;

		for (auto p = transition_probability[i].begin(); p != transition_probability[i].end(); p++) {
			file << p->first << endl << p->second << endl;
		}
	}

	for (int i = 0; i < number_states; i++) {
		for (int j = 0; j < number_gaussian_components; j++) {
			file << GMM[i]->weight[j] << endl;
		}
	}
	for (int i = 0; i < number_states; i++) {
		for (int j = 0; j < number_gaussian_components; j++) {
			for (int k = 0; k < dimension_event; k++) {
				file << GMM[i]->mean[j][k] << endl;
			}
		}
	}
	for (int i = 0; i < number_states; i++) {
		for (int j = 0; j < number_gaussian_components; j++) {
			for (int k = 0; k < dimension_event; k++) {
				if (type_covariance == "diagonal") {
					file << GMM[i]->diagonal_covariance[j][k] << endl;
				}
				else {
					for (int l = 0; l < dimension_event; l++) {
						file << GMM[i]->covariance[j][k][l] << endl;
					}
				}
			}
		}
	}
	file.close();
}

double Continuous_Hidden_Markov_Model::Baum_Welch_Algorithm(int number_events, int length_event[], vector<int> state[], double minimum_variance, double ***_event){
	double log_likelihood = 0;

	double *new_initial_probability = new double[this->number_states];
	unordered_map<int, double> *new_transition_probability[2] = { new unordered_map<int, double>[this->number_states], new unordered_map<int, double>[this->number_states] };

	double ***new_weight = new double**[number_gaussian_components];
	double ****new_mean = new double***[number_gaussian_components];
	double ****new_diagonal_covariance = 0;
	double *****new_covariance = 0;

	for (int i = 0; i < this->number_states; i++){
		new_initial_probability[i] = 0;
	}
	for (int h = 0; h < number_events; h++){
		for (int i = 0; i < state[h].size(); i++){
			for (int j = 0; j < state[h].size(); j++){
				auto p = transition_probability[state[h][i]].find(state[h][j]);

				if (p == transition_probability[state[h][i]].end()){
					transition_probability[state[h][i]].insert(pair<int, double>(state[h][j], 1.0 / this->number_states));
				}
				if ((p = new_transition_probability[0][state[h][i]].find(state[h][j])) == new_transition_probability[0][state[h][i]].end()){
					new_transition_probability[0][state[h][i]].insert(pair<int, double>(state[h][j], 0));
				}
				if ((p = new_transition_probability[1][state[h][i]].find(state[h][j])) == new_transition_probability[1][state[h][i]].end()){
					new_transition_probability[1][state[h][i]].insert(pair<int, double>(state[h][j], 0));
				}
			}
		}
	}

	for (int j = 0; j < number_gaussian_components; j++){
		new_mean[j] = new double**[this->number_states];
		new_weight[j] = new double*[this->number_states];

		for (int i = 0; i < this->number_states; i++){
			new_mean[j][i] = new double*[dimension_event];
			new_weight[j][i] = new double[2];

			for (int k = 0; k < dimension_event; k++){
				new_mean[j][i][k] = new double[2];
				new_mean[j][i][k][0] = 0;
				new_mean[j][i][k][1] = 0;
			}
			new_weight[j][i][0] = 0;
			new_weight[j][i][1] = 0;
		}
	}

	if (type_covariance == "diagonal"){
		new_diagonal_covariance = new double***[number_gaussian_components];

		for (int i = 0; i < number_gaussian_components; i++){
			new_diagonal_covariance[i] = new double**[this->number_states];

			for (int j = 0; j < this->number_states; j++){
				new_diagonal_covariance[i][j] = new double*[dimension_event];

				for (int k = 0; k < dimension_event; k++){
					new_diagonal_covariance[i][j][k] = new double[2];
					new_diagonal_covariance[i][j][k][0] = 0;
					new_diagonal_covariance[i][j][k][1] = 0;
				}
			}
		}
	}
	else{
		new_covariance = new double****[number_gaussian_components];

		for (int j = 0; j < number_gaussian_components; j++){
			new_covariance[j] = new double***[this->number_states];

			for (int i = 0; i < this->number_states; i++){
				new_covariance[j][i] = new double**[dimension_event];

				for (int k = 0; k < dimension_event; k++){
					new_covariance[j][i][k] = new double*[dimension_event];

					for (int l = 0; l < dimension_event; l++){
						new_covariance[j][i][k][l] = new double[2];
						new_covariance[j][i][k][l][0] = 0;
						new_covariance[j][i][k][l][1] = 0;
					}
				}
			}
		}
	}

	#pragma omp parallel for
	for (int h = 0; h < number_events; h++){
		int number_states = state[h].size();

		double **alpha = new double*[length_event[h]];
		double **beta = new double*[length_event[h]];
		double **gamma = new double*[length_event[h]];
		double **likelihood = new double*[length_event[h]];

		double ***gaussian_distribution = new double**[length_event[h]];
		double ***theta = new double**[number_states];

		for (int t = 0; t < length_event[h]; t++){
			alpha[t] = new double[number_states];
			beta[t] = new double[number_states];
			gamma[t] = new double[number_states];
			gaussian_distribution[t] = new double*[number_states];
			likelihood[t] = new double[number_states];

			for (int i = 0; i < number_states; i++){
				gaussian_distribution[t][i] = new double[number_gaussian_components];
			}
		}
		for (int i = 0; i < number_states; i++){
			theta[i] = new double*[number_gaussian_components];

			for (int j = 0; j < number_gaussian_components; j++){
				theta[i][j] = new double[length_event[h]];
			}
		}

		for (int t = 0; t < length_event[h]; t++){
			double sum = 0;

			for (int i = 0; i < number_states; i++){
				int k = state[h][i];

				for (int j = 0; j < number_gaussian_components; j++){
					gaussian_distribution[t][i][j] = GMM[k]->Gaussian_Distribution(_event[h][t], j);
				}
				sum += (likelihood[t][i] = GMM[k]->Calculate_Likelihood(_event[h][t], gaussian_distribution[t][i]));
			}
			for (int i = 0; i < number_states; i++){
				// likelihood[t][i] /= sum;
			}
		}

		#pragma omp atomic
		log_likelihood += Forward_Algorithm(length_event[h], state[h], alpha, likelihood);
		Backward_Algorithm(length_event[h], state[h], beta, likelihood);

		for (int t = 0; t < length_event[h]; t++){
			double sum = 0;
			double sum_theta = 0;

			for (int i = 0; i < number_states; i++){
				sum += (gamma[t][i] = alpha[t][i] * beta[t][i]);
			}
			for (int i = 0; i < number_states; i++){
				gamma[t][i] = (gamma[t][i] == 0) ? (0) : (gamma[t][i] / sum);
			}

			if (t < length_event[h] - 1){
				double sum_delta = 0;

				double **delta = new double*[number_states];

				for (int i = 0; i < number_states; i++){
					delta[i] = new double[number_states];
				}

				for (int i = 0; i < number_states; i++){
					for (int j = 0; j < number_states; j++){
						int k = state[h][i];
						int l = state[h][j];

						if (Access_State(i, j) && state_connection[k].find(l) != state_connection[k].end()) {
							sum_delta += (delta[i][j] = alpha[t][i] * transition_probability[k].find(l)->second * likelihood[t + 1][j] * beta[t + 1][j]);
						}
						else {
							delta[i][j] = 0;
						}
					}
				}
				for (int i = 0; i < number_states; i++){
					for (int j = 0; j < number_states; j++){
						delta[i][j] = (delta[i][j] == 0) ? (0) : (delta[i][j] / sum_delta);
					}
				}

				for (int i = 0; i < number_states; i++){
					for (int j = 0; j < number_states; j++){
						#pragma omp atomic
						new_transition_probability[0][state[h][i]].find(state[h][j])->second += delta[i][j];
					}
				}

				for (int i = 0; i < number_states; i++){
					delete[] delta[i];
				}
				delete[] delta;
			}

			for (int i = 0; i < number_states; i++) {
				int l = state[h][i];

				if (t == 0) {
					for (int j = 0; j < number_gaussian_components; j++) {
						sum_theta += (theta[i][j][t] = (type_model == "ergodic" || i == 0) * initial_probability[l] * GMM[l]->weight[j] * gaussian_distribution[t][i][j] * beta[t][i]);
					}
				}
				else
				if (t >= 1) {
					double *sum = new double[number_gaussian_components];

					memset(sum, 0, sizeof(double) * number_gaussian_components);

					for (int k = 0; k < number_states; k++) {
						int m = state[h][k];

						if (Access_State(k, i) && state_connection[m].find(l) != state_connection[m].end()) {
							for (int j = 0; j < number_gaussian_components; j++) {
								sum[j] += alpha[t - 1][k] * transition_probability[m].find(l)->second;
							}
						}
					}
					for (int j = 0; j < number_gaussian_components; j++) {
						sum_theta += (theta[i][j][t] = sum[j] * GMM[l]->weight[j] * gaussian_distribution[t][i][j] * beta[t][i]);
					}
					delete[] sum;
				}
			}

			for (int i = 0; i < number_states; i++){
				for (int j = 0; j < number_gaussian_components; j++){
					theta[i][j][t] = (theta[i][j][t] == 0) ? (0) : (theta[i][j][t] / sum_theta);
				}
			}
		}

		for (int i = 0; i < this->number_states; i++){
			double sum = 0;

			for (int k = 0; k < number_states; k++){
				if (i == state[h][k]){
					sum += gamma[0][k];
				}
			}
			#pragma omp atomic
			new_initial_probability[i] += sum;
		}

		for (int l = 0; l < number_states; l++){
			double sum = 0;

			for (int t = 0; t < length_event[h] - 1; t++){
				sum += gamma[t][l];
			}
			for (int j = 0; j < number_states; j++){
				#pragma omp atomic
				new_transition_probability[1][state[h][l]].find(state[h][j])->second += sum;
			}
		}

		for (int j = 0; j < number_gaussian_components; j++){
			for (int l = 0; l < number_states; l++){
				double sum[2] = { 0, };

				for (int t = 0; t < length_event[h]; t++){
					sum[0] += theta[l][j][t];
					sum[1] += gamma[t][l];
				}
				#pragma omp atomic
				new_weight[j][state[h][l]][0] += sum[0];
				#pragma omp atomic
				new_weight[j][state[h][l]][1] += sum[1];
			}
		}

		for (int j = 0; j < number_gaussian_components; j++){
			for (int k = 0; k < dimension_event; k++){
				for (int m = 0; m < number_states; m++){
					double sum[2] = { 0, };

					for (int t = 0; t < length_event[h]; t++){
						sum[0] += theta[m][j][t] * _event[h][t][k];
						sum[1] += theta[m][j][t];
					}
					#pragma omp atomic
					new_mean[j][state[h][m]][k][0] += sum[0];
					#pragma omp atomic
					new_mean[j][state[h][m]][k][1] += sum[1];
				}
			}
		}

		for (int j = 0; j < number_gaussian_components; j++){
			for (int k = 0; k < dimension_event; k++){
				if (type_covariance == "diagonal"){
					for (int n = 0; n < number_states; n++){
						double sum[2] = { 0, };

						for (int t = 0; t < length_event[h]; t++){
							sum[0] += theta[n][j][t] * (_event[h][t][k] - GMM[state[h][n]]->mean[j][k]) * (_event[h][t][k] - GMM[state[h][n]]->mean[j][k]);
							sum[1] += theta[n][j][t];
						}
						#pragma omp atomic
						new_diagonal_covariance[j][state[h][n]][k][0] += sum[0];
						#pragma omp atomic
						new_diagonal_covariance[j][state[h][n]][k][1] += sum[1];
					}
				}
				else{
					for (int l = 0; l < dimension_event; l++){
						for (int n = 0; n < number_states; n++){
							double sum[2] = { 0, };

							for (int t = 0; t < length_event[h]; t++){
								sum[0] += theta[n][j][t] * (_event[h][t][k] - GMM[state[h][n]]->mean[j][k]) * (_event[h][t][l] - GMM[state[h][n]]->mean[j][l]);
								sum[1] += theta[n][j][t];
							}
							#pragma omp atomic
							new_covariance[j][state[h][n]][k][l][0] += sum[0];
							#pragma omp atomic
							new_covariance[j][state[h][n]][k][l][1] += sum[1];
						}
					}
				}
			}
		}

		for (int i = 0; i < number_states; i++){
			for (int j = 0; j < number_gaussian_components; j++){
				delete[] theta[i][j];
			}
			delete[] theta[i];
		}
		delete[] theta;

		for (int t = 0; t < length_event[h]; t++){
			for (int i = 0; i < number_states; i++){
				delete[] gaussian_distribution[t][i];
			}
			delete[] alpha[t];
			delete[] beta[t];
			delete[] gamma[t];
			delete[] gaussian_distribution[t];
			delete[] likelihood[t];
		}
		delete[] alpha;
		delete[] beta;
		delete[] gamma;
		delete[] gaussian_distribution;
		delete[] likelihood;
	}

	#pragma omp parallel for
	for (int i = 0; i < this->number_states; i++){
		initial_probability[i] = new_initial_probability[i] / number_events;

		for (auto p = new_transition_probability[0][i].begin(); p != new_transition_probability[0][i].end(); p++){
			transition_probability[i].find(p->first)->second = (p->second == 0) ? (0) : (p->second / new_transition_probability[1][i].find(p->first)->second);
		}
		for (int j = 0; j < number_gaussian_components; j++){
			GMM[i]->weight[j] = (new_weight[j][i][0] == 0) ? (0) : (new_weight[j][i][0] / new_weight[j][i][1]);
		}
		for (int j = 0; j < number_gaussian_components; j++){
			for (int k = 0; k < dimension_event; k++){
				GMM[i]->mean[j][k] = (new_mean[j][i][k][0] == 0) ? (0) : (new_mean[j][i][k][0] / new_mean[j][i][k][1]);
			}
		}
		for (int j = 0; j < number_gaussian_components; j++){
			for (int k = 0; k < dimension_event; k++){
				if (type_covariance == "diagonal"){
					GMM[i]->diagonal_covariance[j][k] = (new_diagonal_covariance[j][i][k][0] == 0) ? (0) : (new_diagonal_covariance[j][i][k][0] / new_diagonal_covariance[j][i][k][1]);

					if (GMM[i]->diagonal_covariance[j][k] < minimum_variance){
						GMM[i]->diagonal_covariance[j][k] = minimum_variance;
					}
				}
				else{
					for (int l = 0; l < dimension_event; l++){
						GMM[i]->covariance[j][k][l] = (new_covariance[j][i][k][l][0] == 0) ? (0) : (new_covariance[j][i][k][l][0] / new_covariance[j][i][k][l][1]);
					}
				}
			}
		}
	}

	for (int i = 0; i < this->number_states; i++){
		valid_transition[i].clear();
	}
	for (int i = 0; i < this->number_states; i++){
		for (auto p = transition_probability[i].begin(); p != transition_probability[i].end(); p++){
			if (p->second){
				valid_transition[p->first].insert(pair<int, double>(i, p->second));
			}
		}
	}

	if (type_covariance == "diagonal"){
		for (int j = 0; j < number_gaussian_components; j++){
			for (int i = 0; i < this->number_states; i++){
				for (int k = 0; k < dimension_event; k++){
					delete[] new_diagonal_covariance[j][i][k];
				}
				delete[] new_diagonal_covariance[j][i];
			}
			delete[] new_diagonal_covariance[j];
		}
		delete[] new_diagonal_covariance;
	}
	else{
		for (int j = 0; j < number_gaussian_components; j++){
			for (int i = 0; i < this->number_states; i++){
				for (int k = 0; k < dimension_event; k++){
					for (int l = 0; l < dimension_event; l++){
						delete[] new_covariance[j][i][k][l];
					}
					delete[] new_covariance[j][i][k];
				}
				delete[] new_covariance[j][i];
			}
			delete[] new_covariance[j];
		}
		delete[] new_covariance;
	}

	for (int j = 0; j < number_gaussian_components; j++){
		for (int i = 0; i < this->number_states; i++){
			for (int k = 0; k < dimension_event; k++){
				delete[] new_mean[j][i][k];
			}
			delete[] new_mean[j][i];
			delete[] new_weight[j][i];
		}
		delete[] new_mean[j];
		delete[] new_weight[j];
	}
	delete[] new_mean;
	delete[] new_weight;

	for (int i = 0; i < this->number_states; i++){
		new_transition_probability[0][i].clear();
		new_transition_probability[1][i].clear();
	}
	delete[] new_initial_probability;
	delete[] new_transition_probability[0];
	delete[] new_transition_probability[1];

	return log_likelihood;
}
double Continuous_Hidden_Markov_Model::Evaluation(int length_event, double **_event){
	double log_likelihood = 0;

	double **alpha = new double*[length_event];

	for (int t = 0; t < length_event; t++){
		alpha[t] = new double[number_states];
	}
	for (int t = 0; t < length_event; t++){
		double scale;
		double sum_alpha = 0;

		if (t == 0){
			for (int i = 0; i < number_states; i++){
				sum_alpha += (alpha[t][i] = initial_probability[i] * GMM[i]->Calculate_Likelihood(_event[t]));
			}
		}
		else
		if (t >= 1){
			for (int i = 0; i < number_states; i++){
				double sum = 0;

				for (auto p = valid_transition[i].begin(); p != valid_transition[i].end(); p++){
					sum += alpha[t - 1][p->first] * p->second;
				}
				sum_alpha += (alpha[t][i] = sum * GMM[i]->Calculate_Likelihood(_event[t]));
			}
		}
		if (sum_alpha == 0){
			log_likelihood = -std::numeric_limits<double>::infinity();
			break;
		}

		scale = 1.0 / sum_alpha;
		for (int i = 0; i < number_states; i++){
			alpha[t][i] *= scale;
		}
		log_likelihood += log(scale);
	}

	for (int t = 0; t < length_event; t++){
		delete[] alpha[t];
	}
	delete[] alpha;

	return log_likelihood;
}
double Continuous_Hidden_Markov_Model::Viterbi_Algorithm(string *optimal_label_sequence, int **optimal_state_sequence, int length_event, double **_event){
	int *state_sequence = new int[length_event];

	double log_likelihood = 0;

	vector<int*>	gamma;
	vector<double*> delta;
	vector<string*> label;

	for (int t = 0; t < length_event; t++){
		double scale;
		double sum = 0;
		double sum_delta = 0;

		double *likelihood = new double[number_states];

		gamma.push_back(new int[number_states]);
		delta.push_back(new double[number_states]);
		label.push_back(new string[number_states]);

		for (int i = 0; i < number_states; i++){
			likelihood[i] = GMM[i]->Calculate_Likelihood(_event[t]);

			sum += likelihood[i];
		}
		if (sum == 0){
			log_likelihood = -std::numeric_limits<double>::infinity();
			delete[] likelihood;
			break;
		}

		for (int i = 0; i < number_states; i++){
			if ((likelihood[i] /= sum) == 0){
				continue;
			}

			if (t == 0){
				delta[t][i] = initial_probability[i] * likelihood[i];
				label[t][i] = state_label[i];
			}
			else{
				int argmax;

				double max = -1;

				for (auto q = valid_transition[i].begin(); q != valid_transition[i].end(); q++){
					double p = delta[t - 1][q->first] * q->second;

					if (max < p){
						argmax = q->first;
						max = p;
					}
				}
				delta[t][i] = max * likelihood[i];
				gamma[t][i] = argmax;

				if (state_label[i] != state_label[argmax]){
					label[t][i] = state_label[i];
				}
			}
			sum_delta += delta[t][i];
		}
		if (sum_delta == 0){
			log_likelihood = -std::numeric_limits<double>::infinity();
			delete[] likelihood;
			break;
		}

		if (t == length_event - 1){
			int argmax;

			double max = -1;

			for (int i = 0; i < number_states; i++){
				if (max < delta[t][i]){
					max = delta[t][argmax = i];
				}
			}
			Search_State_Sequence(t, argmax, state_sequence, gamma);
		}
		log_likelihood -= log(scale = 1 / sum_delta);

		for (int i = 0; i < number_states; i++){
			delta[t][i] *= scale;
		}
		delete[] likelihood;
	}

	if (_finite(log_likelihood)){
		if (optimal_label_sequence){
			string label_sequence;

			for (int t = 0; t < length_event; t++){
				if (!label[t][state_sequence[t]].empty()){
					if (!label_sequence.empty()){
						label_sequence += ' ';
					}
					label_sequence += label[t][state_sequence[t]];
				}
			}
			(*optimal_label_sequence) = label_sequence;
		}
		if (optimal_state_sequence){
			(*optimal_state_sequence) = new int[length_event];

			for (int t = 0; t < length_event; t++){
				(*optimal_state_sequence)[t] = state_sequence[t];
			}
		}
	}

	for (int t = 0; t < length_event; t++){
		delete[] delta[t];
		delete[] gamma[t];
		delete[] label[t];
	}
	delete[] state_sequence;

	delta.clear();
	gamma.clear();
	label.clear();

	return log_likelihood;
}
