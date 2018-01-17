#include <hash_map>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "CHMM.h"

using namespace stdext;

void Read_MNIST(char training_set_images[], char training_set_labels[], char test_set_images[], char test_set_labels[], int time_step, int number_training, int number_test, int *label, double ***input){
	FILE *file;

	if (file = fopen(training_set_images, "rb")){
		for (int h = 0, value; h < 4; h++){
			fread(&value, sizeof(int), 1, file);
		}
		for (int h = 0; h < number_training; h++){
			unsigned char pixel;

			for (int j = 0; j < time_step; j++){
				for (int k = 0; k < 28 * 28 / time_step; k++){
					fread(&pixel, sizeof(unsigned char), 1, file);
					input[h][j][k] = pixel / 255.0;
				}
			}
		}
		fclose(file);
	}
	else{
		fprintf(stderr, "[Read_MNIST], %s not found\n", training_set_images);
	}

	if (file = fopen(training_set_labels, "rb")){
		for (int h = 0, value; h < 2; h++){
			fread(&value, sizeof(int), 1, file);
		}
		for (int h = 0; h < number_training; h++){
			unsigned char value;

			fread(&value, sizeof(unsigned char), 1, file);
			label[h] = value;
		}
		fclose(file);
	}
	else{
		fprintf(stderr, "[Read_MNIST], %s not found\n", training_set_labels);
	}

	if (file = fopen(test_set_images, "rb")){
		for (int h = 0, value; h < 4; h++){
			fread(&value, sizeof(int), 1, file);
		}
		for (int h = number_training; h < number_training + number_test; h++){
			unsigned char pixel;

			for (int j = 0; j < time_step; j++){
				for (int k = 0; k < 28 * 28 / time_step; k++){
					fread(&pixel, sizeof(unsigned char), 1, file);
					input[h][j][k] = pixel / 255.0;
				}
			}
		}
		fclose(file);
	}
	else{
		fprintf(stderr, "[Read_MNIST], %s not found\n", test_set_images);
	}

	if (file = fopen(test_set_labels, "rb")){
		for (int h = 0, value; h < 2; h++){
			fread(&value, sizeof(int), 1, file);
		}
		for (int h = number_training; h < number_training + number_test; h++){
			unsigned char value;

			fread(&value, sizeof(unsigned char), 1, file);
			label[h] = value;
		}
		fclose(file);
	}
	else{
		fprintf(stderr, "[Read_MNIST], %s not found\n", test_set_labels);
	}
}

int main(){
	char *state_label[] = { "0", "0", "0", "1", "1", "1", "2", "2", "2", "3", "3", "3", "4", "4", "4", "5", "5", "5", "6", "6", "6", "7", "7", "7", "8", "8", "8", "9", "9", "9", "-", "-", "-" };
	char type_covariance[] = "diagonal";	// <-> "full"
	char type_model[] = "bakis";		// <-> "ergodic"

	int dimension_event = 56;
	int length_data = 14;
	int number_iterations = 100;
	int number_threads = 4;
	int number_training = 60000;
	int number_test = 10000;

	int number_gaussian_components = 1;

	double minimum_variance = 0.05; // prevents overfitting in the case of diagonal covariance

	int *label = new int[number_training + number_test];
	int *length_event = new int[number_training + number_test];

	double ***_event = new double**[number_training + number_test];

	Continuous_Hidden_Markov_Model *CHMM;

	for (int i = 0; i < number_training + number_test; i++){
		_event[i] = new double*[length_event[i] = length_data];

		for (int j = 0; j < length_event[i]; j++){
			_event[i][j] = new double[dimension_event];
		}
	}
	Read_MNIST("train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", length_data, number_training, number_test, label, _event);
	omp_set_num_threads(number_threads);

	// Construct CHMM
	{
		hash_map<int, bool> *state_connection = new hash_map<int, bool>[sizeof(state_label) / sizeof(state_label[0])];

		int number_states = sizeof(state_label) / sizeof(state_label[0]);

		for (int i = 0; i < number_states; i++){
			for (int j = 0; j < number_states; j++){
				// for ergodic model
				// state_connection[i].insert(hash_map<int, bool>::value_type(j, true));

				// for bakis model
				if (i == j || (i % 3 == 2 && j % 3 == 0) || j - i == 1){
					state_connection[i].insert(hash_map<int, bool>::value_type(j, true));
				}
			}
		}
		CHMM = new Continuous_Hidden_Markov_Model(state_connection, type_covariance, type_model, state_label, dimension_event, number_gaussian_components, number_states);

		for (int i = 0; i < number_states; i++){
			state_connection[i].clear();
		}
		delete[] state_connection;
	}

	// Initialize and Train CHMM
	{
		int *number_states = new int[number_training];

		int **state = new int*[number_training];

		for (int i = 0; i < number_training; i++){
			state[i] = new int[number_states[i] = 9];

			for (int j = 0; j < number_states[i]; j++){
				state[i][j] = ((j / 3 == 0 || j / 3 == 2) ? (30) : (3 * label[i])) + (j % 3);
			}
		}
		CHMM->Initialize(number_training, length_event, _event);

		for (int h = 0, time = clock(); h < number_iterations; h++){
			int number_correct[2] = { 0, };

			double log_likelihood = CHMM->Baum_Welch_Algorithm(number_training, length_event, number_states, state, minimum_variance, _event);

			#pragma omp parallel for
			for (int i = 0; i < number_training + number_test; i++){
				char *optimal_label_sequence;

				int *optimal_state_sequence;

				CHMM->Viterbi_Algorithm(&optimal_label_sequence, &optimal_state_sequence, length_event[i], _event[i]);

				/*for(int t = 0;t < length_event[i];t++){
				printf("%d ", optimal_state_sequence[t]);
				}
				printf("\n");*/

				// printf("%s\n", optimal_label_sequence);

				#pragma omp atomic
				number_correct[(i < number_training) ? (0) : (1)] += (atoi(&optimal_label_sequence[2]) == label[i]);

				if (optimal_label_sequence) delete[] optimal_label_sequence;
				if (optimal_state_sequence) delete[] optimal_state_sequence;
			}
			printf("score: %d / %d, %d / %d  L: %lf  step %d  %.2lf sec\n", number_correct[0], number_training, number_correct[1], number_test, log_likelihood, h + 1, (double)(clock() - time) / CLOCKS_PER_SEC);
		}

		for (int i = 0; i < number_training; i++){
			delete[] state[i];
		}
		delete[] number_states;
		delete[] state;
	}

	delete CHMM;
	return 0;
}
