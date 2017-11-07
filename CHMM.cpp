#include <float.h>
#include <limits>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CHMM.h"
#include "Kmeans.h"

void Continuous_Hidden_Markov_Model::Search_State_Sequence(int event_index, int state_index, int state_sequence[], int **gamma){		
	state_sequence[event_index] = state_index;
	for(int t = event_index - 1;t >= 0;t--){
		state_sequence[t] = gamma[t + 1][state_sequence[t + 1]];
	}
}

bool Continuous_Hidden_Markov_Model::Access_State(int previous_state_index, int state_index){
	int i = previous_state_index;
	int j = state_index;

	return ((!strcmp(type_model, "ergodic") || (j - i == 0) || (j - i == 1)));
}

double Continuous_Hidden_Markov_Model::Backward_Algorithm(int length_event, int number_states, int state[], double scale[], double **beta, double **likelihood){
	double log_likelihood = 0;

	for(int t = length_event - 1;t >= 0;t--){
		if(t == length_event - 1){
			for(int i = 0;i < number_states;i++){
				beta[t][i] = (!strcmp(type_model, "ergodic") || i == number_states - 1);
			}
		}
		else
		if(t <= length_event - 2){
			for(int i = 0;i < number_states;i++){
				double sum = 0;

				for(int j = 0;j < number_states;j++){
					int k = state[i];
					int l = state[j];

					if(state_connection[k][l] && Access_State(i, j)){
						sum += transition_probability[k][l] * likelihood[t + 1][j] * beta[t + 1][j];
					}
				}
				beta[t][i] = sum;
			}
		}
		for(int i = 0;i < number_states;i++){
			beta[t][i] *= scale[t];
		}
		// log_likelihood += log(scale[t]);
	}
	return -log_likelihood;
}
double Continuous_Hidden_Markov_Model::Forward_Algorithm(int length_event, int number_states, int state[], double scale[], double **alpha, double **likelihood){
	double log_likelihood = 0;
		
	for(int t = 0;t < length_event;t++){
		double sum_alpha = 0;

		if(t == 0){
			for(int i = 0;i < number_states;i++){
				int j = state[i];
					
				sum_alpha += (alpha[t][i] = (!strcmp(type_model, "ergodic") || i == 0) * initial_probability[j] * likelihood[t][i]);
			}
		}
		else
		if(t >= 1){
			for(int i = 0;i < number_states;i++){
				double sum = 0;

				for(int j = 0;j < number_states;j++){
					int k = state[i];
					int l = state[j];

					if(state_connection[l][k] && Access_State(j, i)){
						sum += alpha[t - 1][j] * transition_probability[l][k];
					}
				}
				sum_alpha += (alpha[t][i] = sum * likelihood[t][i]);
			}
		}

		// scale
		if(!_finite(log(scale[t] = 1.0 / sum_alpha)) || _isnan(log(scale[t]))){
			fprintf(stderr, "[Forward Algorithm] [scale[%d]: %lf]\n", t, scale[t]);
		}
		for(int i = 0;i < number_states;i++){
			alpha[t][i] *= scale[t];
		}
		log_likelihood += log(scale[t]);
	}
	return -log_likelihood;
}

Continuous_Hidden_Markov_Model::Continuous_Hidden_Markov_Model(bool **state_connection, char type_covariance[], char type_model[], char **state_label, int dimension_event, int number_gaussian_components, int number_states){
	strcpy(this->type_covariance, type_covariance);
	strcpy(this->type_model, type_model);
	this->dimension_event			 = dimension_event;
	this->maximum_length_label		 = 0;
	this->number_gaussian_components = number_gaussian_components;
	this->number_states				 = number_states;
	this->state_connection			 = new bool*[number_states];
	this->state_label				 = new char*[number_states];
	
	for(int i = 0;i < number_states;i++){
		this->state_connection[i] = new bool[number_states];

		if(state_connection){
			for(int j = 0;j < number_states;j++){
				this->state_connection[i][j] = state_connection[i][j];
			}
		}
	}
	for(int i = 0;i < number_states;i++){
		int length_label = strlen(state_label[i]) + 1;

		if(maximum_length_label < length_label){
			maximum_length_label = length_label;
		}
		this->state_label[i] = new char[length_label];
		strcpy(this->state_label[i], state_label[i]);
	}

	initial_probability		= new double[number_states];
	transition_probability	= new double*[number_states];
	GMM						= new Gaussian_Mixture_Model*[number_states];
	for(int i = 0;i < number_states;i++){
		transition_probability[i]	= new double[number_states];
		GMM[i]						= new Gaussian_Mixture_Model(type_covariance, dimension_event, number_gaussian_components);
	}
}
Continuous_Hidden_Markov_Model::~Continuous_Hidden_Markov_Model(){
	for(int i = 0;i < number_states;i++){
		delete[] state_connection[i];
		delete[] state_label[i];
	}
	delete[] state_connection;
	delete[] state_label;
		
	for(int i = 0;i < number_states;i++){
		delete[] transition_probability[i];
		delete GMM[i];
	}
	delete[] initial_probability;
	delete[] transition_probability;
	delete GMM;
}

void Continuous_Hidden_Markov_Model::Initialize(int number_events, int length_event[], double ***_event, int seed){
	int number_data = 0;

	double sum = 0;

	double **data;

	KMeans kmeans = KMeans(dimension_event, number_gaussian_components);

	srand(seed);

	for(int i = 0;i < number_states;i++){
		sum += (initial_probability[i] = (rand() + 1.0) / RAND_MAX);
	}
	for(int i = 0;i < this->number_states;i++){
		initial_probability[i] /= sum;
	}

	for(int i = 0;i < this->number_states;i++){
		double sum = 0;

		for(int j = 0;j < this->number_states;j++){
			sum += (transition_probability[i][j] = (rand() + 1.0) / RAND_MAX);
		}
		for(int j = 0;j < this->number_states;j++){
			transition_probability[i][j] /= sum;
		}
	}

	for(int i = 0;i < number_events;i++){
		number_data += length_event[i];
	}
	data = new double*[number_data];

	for(int i = 0, index = 0;i < number_events;i++){
		for(int j = 0;j < length_event[i];j++){
			data[index] = new double[dimension_event];

			for(int k = 0;k < dimension_event;k++){
				data[index][k] = _event[i][j][k];
			}
			index++;
		}
	}

	kmeans.Initialize(number_data, data);
	while(kmeans.Cluster(number_data, data));
	
	for(int i = 0;i < this->number_states;i++){
		for(int j = 0;j < number_gaussian_components;j++){
			for(int k = 0;k < dimension_event;k++){
				if(!strcmp(type_covariance, "diagonal")){
					GMM[i]->diagonal_covariance[j][k] = 1;
				}
				else{
					for(int l = 0;l < dimension_event;l++){
						GMM[i]->covariance[j][k][l] = (k == l);
					}
				}
				GMM[i]->mean[j][k] = kmeans.centroid[j][k];
			}
			GMM[i]->weight[j] = 1.0 / number_gaussian_components;
		}
	}

	for(int i = 0;i < number_data;i++){
		delete[] data[i];
	}
	delete[] data;
}
void Continuous_Hidden_Markov_Model::Load_Parameter(char path[]){
	FILE *file = fopen(path, "rt");

	if(file){
		for(int i = 0;i < number_states;i++){
			fscanf(file, "%lf", &initial_probability[i]);
		}
		for(int i = 0;i < number_states;i++){
			for(int j = 0;j < number_states;j++){
				fscanf(file, "%lf", &transition_probability[i][j]);
			}
		}
		for(int i = 0;i < number_states;i++){
			for(int j = 0;j < number_gaussian_components;j++){
				fscanf(file, "%lf", &(GMM[i]->weight[j]));
			}
		}
		for(int i = 0;i < number_states;i++){
			for(int j = 0;j < number_gaussian_components;j++){
				for(int k = 0;k < dimension_event;k++){
					fscanf(file, "%lf", &(GMM[i]->mean[j][k]));
				}
			}
		}
		for(int i = 0;i < number_states;i++){
			for(int j = 0;j < number_gaussian_components;j++){
				for(int k = 0;k < dimension_event;k++){	
					for(int l = 0;l < dimension_event;l++){
						fscanf(file, "%lf", &(GMM[i]->covariance[j][k][l]));
					}
				}
			}
		}
		printf("Parameter Loaded\n");
		fclose(file);
	}
	else{
		fprintf(stderr, "[Load Parameter], [%s is not found]\n", path);
	}
}
void Continuous_Hidden_Markov_Model::Save_Parameter(char path[]){
	FILE *file = fopen(path, "wt");

	for(int i = 0;i < number_states;i++){
		fprintf(file, "%f\n", initial_probability[i]);
	}
	for(int i = 0;i < number_states;i++){
		for(int j = 0;j < number_states;j++){
			fprintf(file, "%f\n", transition_probability[i][j]);
		}
	}		
	for(int i = 0;i < number_states;i++){
		for(int j = 0;j < number_gaussian_components;j++){
			fprintf(file, "%f\n", GMM[i]->weight[j]);
		}
	}		
	for(int i = 0;i < number_states;i++){
		for(int j = 0;j < number_gaussian_components;j++){
			for(int k = 0;k < dimension_event;k++){
				fprintf(file, "%f\n", GMM[i]->mean[j][k]);
			}
		}
	}		
	for(int i = 0;i < number_states;i++){
		for(int j = 0;j < number_gaussian_components;j++){
			for(int k = 0;k < dimension_event;k++){	
				for(int l = 0;l < dimension_event;l++){
					fprintf(file, "%f\n", GMM[i]->covariance[j][k][l]);
				}
			}
		}
	}
	printf("Parameter Saved\n");
	fclose(file);
}

double Continuous_Hidden_Markov_Model::Baum_Welch_Algorithm(int number_events, int length_event[], int number_states[], int **state, double minimum_variance, double ***_event){
	double log_likelihood = 0;

	double *new_initial_probability		 = new double[this->number_states];
	double ***new_transition_probability = new double**[this->number_states];

	double ***new_weight				= new double**[number_gaussian_components];
	double ****new_mean					= new double***[number_gaussian_components];
	double ****new_diagonal_covariance;
	double *****new_covariance;

	for(int i = 0;i < this->number_states;i++){
		new_transition_probability[i] = new double*[this->number_states];

		for(int j = 0;j < this->number_states;j++){
			new_transition_probability[i][j] = new double[2];
			new_transition_probability[i][j][0] = 0;
			new_transition_probability[i][j][1] = 0;
		}
		new_initial_probability[i] = 0;
	}

	for(int j = 0;j < number_gaussian_components;j++){
		new_mean[j]		= new double**[this->number_states];
		new_weight[j]	= new double*[this->number_states];

		for(int i = 0;i < this->number_states;i++){
			new_mean[j][i]	 = new double*[dimension_event];
			new_weight[j][i] = new double[2];

			for(int k = 0;k < dimension_event;k++){
				new_mean[j][i][k]	 = new double[2];	
				new_mean[j][i][k][0] = 0;
				new_mean[j][i][k][1] = 0;
			}
			new_weight[j][i][0] = 0;
			new_weight[j][i][1] = 0;
		}
	}

	if(!strcmp(type_covariance, "diagonal")){
		new_diagonal_covariance = new double***[number_gaussian_components];

		for(int i = 0;i < number_gaussian_components;i++){
			new_diagonal_covariance[i] = new double**[this->number_states];

			for(int j = 0;j < this->number_states;j++){
				new_diagonal_covariance[i][j] = new double*[dimension_event];

				for(int k = 0;k < dimension_event;k++){
					new_diagonal_covariance[i][j][k] = new double[2];
					new_diagonal_covariance[i][j][k][0] = 0;
					new_diagonal_covariance[i][j][k][1] = 0;
				}
			}
		}
	}
	else{
		new_covariance = new double****[number_gaussian_components];

		for(int j = 0;j < number_gaussian_components;j++){
			new_covariance[j] = new double***[this->number_states];

			for(int i = 0;i < this->number_states;i++){
				new_covariance[j][i] = new double**[dimension_event];

				for(int k = 0;k < dimension_event;k++){
					new_covariance[j][i][k] = new double*[dimension_event];

					for(int l = 0;l < dimension_event;l++){
						new_covariance[j][i][k][l] = new double[2];
						new_covariance[j][i][k][l][0] = 0;
						new_covariance[j][i][k][l][1] = 0;
					}
				}
			}
		}
	}

	#pragma omp parallel for
	for(int h = 0;h < number_events;h++){
		double *scale = new double[length_event[h]];
					
		double **alpha		= new double*[length_event[h]];
		double **beta		= new double*[length_event[h]];
		double **gamma		= new double*[length_event[h]];
		double **likelihood	= new double*[length_event[h]];
				
		double ***delta					= new double**[length_event[h]];
		double ***gaussian_distribution = new double**[length_event[h]];
		double ***theta					= new double**[length_event[h]];

		for(int t = 0;t < length_event[h];t++){
			alpha[t]				 = new double[number_states[h]];
			beta[t]					 = new double[number_states[h]];
			delta[t]				 = new double*[number_states[h]];
			gamma[t]				 = new double[number_states[h]];
			gaussian_distribution[t] = new double*[number_states[h]];
			likelihood[t]			 = new double[number_states[h]];
			theta[t]				 = new double*[number_states[h]];
					
			for(int i = 0;i < number_states[h];i++){
				delta[t][i]					= new double[number_states[h]];
				gaussian_distribution[t][i] = new double[number_gaussian_components];
				theta[t][i]					= new double[number_gaussian_components];
			}
		}

		for(int t = 0;t < length_event[h];t++){
			for(int i = 0;i < number_states[h];i++){
				int k = state[h][i];

				for(int j = 0;j < number_gaussian_components;j++){
					gaussian_distribution[t][i][j] = GMM[k]->Gaussian_Distribution(_event[h][t], j);
				}
				likelihood[t][i] = GMM[k]->Calculate_Likelihood(_event[h][t], gaussian_distribution[t][i]);
			}
		}

		#pragma omp atomic
		log_likelihood +=	Forward_Algorithm(length_event[h], number_states[h], state[h], scale, alpha, likelihood);
							Backward_Algorithm(length_event[h], number_states[h], state[h], scale, beta, likelihood);

		for(int t = 0;t < length_event[h];t++){
			double sum		 = 0;
			double sum_theta = 0;
					
			for(int i = 0;i < number_states[h];i++){
				sum += (gamma[t][i] = alpha[t][i] * beta[t][i]);
			}
			for(int i = 0;i < number_states[h];i++){
				gamma[t][i] /= sum;
			}

			if(t < length_event[h] - 1){
				double sum = 0;

				for(int i = 0;i < number_states[h];i++){							
					for(int j = 0;j < number_states[h];j++){
						int k = state[h][i];
						int l = state[h][j];
						
						sum += (delta[t][i][j] = (state_connection[k][l] && Access_State(i, j)) * alpha[t][i] * transition_probability[k][l] * likelihood[t + 1][j] * beta[t + 1][j]);
					}
				}
				for(int i = 0;i < number_states[h];i++){
					for(int j = 0;j < number_states[h];j++){
						delta[t][i][j] /= sum;
					}
				}
			}
					
			for(int i = 0;i < number_states[h];i++){
				int l = state[h][i];
						
				for(int j = 0;j < number_gaussian_components;j++){
					double sum = 0;
							
					if(t == 0){
						sum = (!strcmp(type_model, "ergodic") || i == 0) * initial_probability[l];
					}
					else
					if(t >= 1){
						for(int k = 0;k < number_states[h];k++){
							int m = state[h][k];

							if(state_connection[m][l] && Access_State(k, i)){
								sum += alpha[t - 1][k] * transition_probability[m][l];
							}
						}
					}
					sum_theta += (theta[t][i][j] = sum * GMM[l]->weight[j] * gaussian_distribution[t][i][j] * beta[t][i]);
				}
			}

			for(int i = 0;i < number_states[h];i++){
				for(int j = 0;j < number_gaussian_components;j++){
					theta[t][i][j] /= sum_theta;
				}
			}
		}

		for(int i = 0;i < this->number_states;i++){
			double sum = 0;

			for(int k = 0;k < number_states[h];k++){
				if(i == state[h][k]){
					sum += gamma[0][k];
				}
			}
			#pragma omp atomic
			new_initial_probability[i] += sum;

			for(int j = 0;j < this->number_states;j++){
				double sum[2] = {0, };

				for(int t = 0;t < length_event[h] - 1;t++){
					for(int l = 0;l < number_states[h];l++){
						if(i == state[h][l]){
							for(int m = 0;m < number_states[h];m++){
								if(j == state[h][m]){
									sum[0] += delta[t][l][m];
								}
							}
							sum[1] += gamma[t][l];
						}
					}
				}
				#pragma omp atomic
				new_transition_probability[i][j][0] += sum[0];
				#pragma omp atomic
				new_transition_probability[i][j][1] += sum[1];
			}

			for(int j = 0;j < number_gaussian_components;j++){
				double sum[2] = {0, };

				for(int t = 0;t < length_event[h];t++){
					for(int l = 0;l < number_states[h];l++){
						if(i == state[h][l]){
							sum[0] += theta[t][l][j];
							sum[1] += gamma[t][l];
						}
					}
				}
				#pragma omp atomic
				new_weight[j][i][0] += sum[0];
				#pragma omp atomic
				new_weight[j][i][1] += sum[1];
			}
			
			for(int j = 0;j < number_gaussian_components;j++){
				for(int k = 0;k < dimension_event;k++){
					double sum[2] = {0, };

					for(int t = 0;t < length_event[h];t++){
						for(int m = 0;m < number_states[h];m++){
							if(i == state[h][m]){
								sum[0] += theta[t][m][j] * _event[h][t][k];
								sum[1] += theta[t][m][j];
							}
						}
					}
					#pragma omp atomic
					new_mean[j][i][k][0] += sum[0];
					#pragma omp atomic
					new_mean[j][i][k][1] += sum[1];
				}
			}

			for(int j = 0;j < number_gaussian_components;j++){
				for(int k = 0;k < dimension_event;k++){
					if(!strcmp(type_covariance, "diagonal")){
						double sum[2] = {0, };

						for(int t = 0;t < length_event[h];t++){
							for(int n = 0;n < number_states[h];n++){
								if(i == state[h][n]){
									sum[0] += theta[t][n][j] * (_event[h][t][k] - GMM[i]->mean[j][k]) * (_event[h][t][k] - GMM[i]->mean[j][k]);
									sum[1] += theta[t][n][j];
								}
							}
						}
						#pragma omp atomic
						new_diagonal_covariance[j][i][k][0] += sum[0];
						#pragma omp atomic
						new_diagonal_covariance[j][i][k][1] += sum[1];
					}
					else{
						for(int l = 0;l < dimension_event;l++){
							double sum[2] = {0, };
							
							for(int t = 0;t < length_event[h];t++){
								for(int n = 0;n < number_states[h];n++){
									if(i == state[h][n]){
										sum[0] += theta[t][n][j] * (_event[h][t][k] - GMM[i]->mean[j][k]) * (_event[h][t][l] - GMM[i]->mean[j][l]);
										sum[1] += theta[t][n][j];
									}
								}
							}
							#pragma omp atomic
							new_covariance[j][i][k][l][0] += sum[0];
							#pragma omp atomic
							new_covariance[j][i][k][l][1] += sum[1];
						}
					}
				}
			}
		}
				
		for(int t = 0;t < length_event[h];t++){
			for(int i = 0;i < number_states[h];i++){
				delete[] delta[t][i];
				delete[] gaussian_distribution[t][i];
				delete[] theta[t][i];
			}
			delete[] alpha[t];
			delete[] beta[t];
			delete[] delta[t];
			delete[] gamma[t];
			delete[] gaussian_distribution[t];
			delete[] likelihood[t];
			delete[] theta[t];
		}
		delete[] alpha;
		delete[] beta;
		delete[] delta;
		delete[] gamma;
		delete[] gaussian_distribution;
		delete[] likelihood;
		delete[] scale;
		delete[] theta;
	}

	#pragma omp parallel for
	for(int i = 0;i < this->number_states;i++){
		initial_probability[i] = new_initial_probability[i] / number_events;

		for(int j = 0;j < this->number_states;j++){
			transition_probability[i][j] = (new_transition_probability[i][j][0] == 0) ? (0):(new_transition_probability[i][j][0] / new_transition_probability[i][j][1]);
		}
		for(int j = 0;j < number_gaussian_components;j++){
			GMM[i]->weight[j] = (new_weight[j][i][0] == 0) ? (0):(new_weight[j][i][0] / new_weight[j][i][1]);
		}
		for(int j = 0;j < number_gaussian_components;j++){
			for(int k = 0;k < dimension_event;k++){
				GMM[i]->mean[j][k] = (new_mean[j][i][k][0] == 0) ? (0):(new_mean[j][i][k][0] / new_mean[j][i][k][1]);
			}
		}
		for(int j = 0;j < number_gaussian_components;j++){
			for(int k = 0;k < dimension_event;k++){
				if(!strcmp(type_covariance, "diagonal")){
					GMM[i]->diagonal_covariance[j][k] = (new_diagonal_covariance[j][i][k][0] == 0) ? (0):(new_diagonal_covariance[j][i][k][0] / new_diagonal_covariance[j][i][k][1]);

					if(GMM[i]->diagonal_covariance[j][k] < minimum_variance){
						GMM[i]->diagonal_covariance[j][k] = minimum_variance;
					}
				}
				else{
					for(int l = 0;l < dimension_event;l++){					
						GMM[i]->covariance[j][k][k] = (new_covariance[j][i][k][k][0] == 0) ? (0):(new_covariance[j][i][k][k][0] / new_covariance[j][i][k][k][1]);
					}
				}
			}
		}
	}

	if(!strcmp(type_covariance, "diagonal")){
		for(int j = 0;j < number_gaussian_components;j++){
			for(int i = 0;i < this->number_states;i++){
				for(int k = 0;k < dimension_event;k++){
					delete[] new_diagonal_covariance[j][i][k];
				}
				delete[] new_diagonal_covariance[j][i];
			}
			delete[] new_diagonal_covariance[j];
		}
		delete[] new_diagonal_covariance;
	}
	else{
		for(int j = 0;j < number_gaussian_components;j++){
			for(int i = 0;i < this->number_states;i++){
				for(int k = 0;k < dimension_event;k++){
					for(int l = 0;l < dimension_event;l++){
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

	for(int j = 0;j < number_gaussian_components;j++){
		for(int i = 0;i < this->number_states;i++){
			for(int k = 0;k < dimension_event;k++){
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

	for(int i = 0;i < this->number_states;i++){
		for(int j = 0;j < this->number_states;j++){
			delete[] new_transition_probability[i][j];
		}
		delete[] new_transition_probability[i];
	}
	delete[] new_initial_probability;
	delete[] new_transition_probability;

	return log_likelihood;
}
double Continuous_Hidden_Markov_Model::Evaluation(int length_event, double **_event){
	double log_likelihood = 0;

	double **alpha = new double*[length_event];

	for(int t = 0;t < length_event;t++){
		alpha[t] = new double[number_states];
	}
	for(int t = 0;t < length_event;t++){
		double scale;
		double sum_alpha = 0;

		if(t == 0){
			for(int i = 0;i < number_states;i++){					
				sum_alpha += (alpha[t][i] = initial_probability[i] * GMM[i]->Calculate_Likelihood(_event[t]));
			}
		}
		else
		if(t >= 1){
			for(int i = 0;i < number_states;i++){
				double sum = 0;

				for(int j = 0;j < number_states;j++){
					sum += alpha[t - 1][j] * transition_probability[j][i];
				}
				sum_alpha += (alpha[t][i] = sum * GMM[i]->Calculate_Likelihood(_event[t]));
			}
		}
		if(sum_alpha == 0){
			log_likelihood = -std::numeric_limits<double>::infinity();
			break;
		}

		scale = 1.0 / sum_alpha;
		for(int i = 0;i < number_states;i++){
			alpha[t][i] *= scale;
		}
		log_likelihood += log(scale);
	}

	for(int t = 0;t < length_event;t++){
		delete[] alpha[t];
	}
	delete[] alpha;

	return log_likelihood;
}
double Continuous_Hidden_Markov_Model::Viterbi_Algorithm(char optimal_label_sequence[], int optimal_state_sequence[], int length_event, double **_event){
	char ***label = new char**[length_event];
		
	int *state_sequence = new int[length_event];
		
	int **gamma = new int*[length_event];

	double log_likelihood = 0;
		
	double **delta = new double*[length_event];
		
	for(int t = 0;t < length_event;t++){
		delta[t] = new double[number_states];
		gamma[t] = new int[number_states];
		label[t] = new char*[number_states];		
			
		for(int i = 0;i < number_states;i++){
			strcpy(label[t][i] = new char[maximum_length_label], "");
		}
	}

	for(int t = 0;t < length_event;t++){			
		double scale;
		double sum		 = 0;
		double sum_delta = 0;
			
		double *likelihood	= new double[number_states];
		
		for(int i = 0;i < number_states;i++){
			likelihood[i] = GMM[i]->Calculate_Likelihood(_event[t]);

			sum += likelihood[i];
		}
		if(sum == 0){
			log_likelihood = -std::numeric_limits<double>::infinity();
			delete[] likelihood;
			break;
		}

		for(int i = 0;i < number_states;i++){
			likelihood[i] /= sum;

			if(t == 0){
				delta[t][i] = initial_probability[i] * likelihood[i];
			}
			else{
				int argmax;

				double max = -1;

				for(int k = 0;k < number_states;k++){
					double p = delta[t - 1][k] * transition_probability[k][i];

					if(max < p){
						argmax	= k;
						max		= p;
					}
				}
				delta[t][i] = max * likelihood[i];
				gamma[t][i] = argmax;

				if(strcmp(state_label[i], state_label[argmax])){
					strcpy(label[t - 1][argmax], state_label[argmax]);
				}
			}
			sum_delta += delta[t][i];
		}
		if(sum_delta == 0){
			log_likelihood = -std::numeric_limits<double>::infinity();
			delete[] likelihood;
			break;
		}

		if(t == length_event - 1){
			int argmax;

			double max = -1;

			for(int i = 0;i < number_states;i++){
				if(max < delta[t][i]){
					max = delta[t][argmax = i];
				}
			}
			Search_State_Sequence(t, argmax, state_sequence, gamma);
			strcpy(label[t - 1][argmax], state_label[argmax]);
		}
		log_likelihood -= log(scale = 1 / sum_delta);

		for(int i = 0;i < number_states;i++){
			delta[t][i] *= scale;
		}
		delete[] likelihood;
	}

	if(_finite(log_likelihood)){
		char *recent_label = 0;

		if(optimal_label_sequence) strcpy(optimal_label_sequence, "");

		for(int t = 0;t < length_event;t++){
			if(optimal_label_sequence && strcmp(label[t][state_sequence[t]], "") && (recent_label == 0 || strcmp(label[t][state_sequence[t]], recent_label))){
				recent_label = label[t][state_sequence[t]];
				strcat(optimal_label_sequence, label[t][state_sequence[t]]);
				strcat(optimal_label_sequence, " ");
			}
			if(optimal_state_sequence){
				optimal_state_sequence[t] = state_sequence[t];
			}
		}
	}

	for(int t = 0;t < length_event;t++){
		for(int i = 0;i < number_states;i++){
			delete[] label[t][i];
		}
		delete[] delta[t];
		delete[] gamma[t];
		delete[] label[t];
	}
	delete[] delta;
	delete[] gamma;
	delete[] label;
	delete[] state_sequence;

	return log_likelihood;
}
