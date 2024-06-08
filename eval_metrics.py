import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, spearmanr, kendalltau

# Load the JSON file
with open('manual_scored_evalg.json', 'r') as file:
    data = json.load(file)

# Initialize lists to store each kind of score
coherency_scores = []
fluency_scores = []
coherency_score_values = []
perplexity_values = []
fluency_MistralEval_scores = []
coherency_MistralEval_scores = []
fluency_GPTEval_scores = []
coherency_GPTEval_scores = []
fluency_GPT4Eval_scores = []
coherency_GPT4Eval_scores = []
fluency_LlamaEval_scores = []
coherency_LlamaEval_scores = []
fluency_PhiEval_scores = []
coherency_PhiEval_scores = []

# Iterate through the data and collect scores
for entry in data:
    for utterance in entry['utterances']:
        coherency_scores.extend(utterance['coherency'])
        fluency_scores.extend(utterance['fluency'])
        coherency_score_values.extend(utterance['coherency_score'])
        perplexity_values.extend(utterance['preplexity'])
        fluency_MistralEval_scores.extend(utterance['fluency_MistralEval'])
        coherency_MistralEval_scores.extend(utterance['coherency_MistralEval'])
        fluency_GPTEval_scores.extend(utterance['fluency_GPTEval'])
        coherency_GPTEval_scores.extend(utterance['coherency_GPTEval'])
        fluency_GPT4Eval_scores.extend(utterance['fluency_GPT4Eval'])
        coherency_GPT4Eval_scores.extend(utterance['coherency_GPT4Eval'])
        fluency_LlamaEval_scores.extend(utterance['fluency_LlamaEval'])
        coherency_LlamaEval_scores.extend(utterance['coherency_LlamaEval'])
        fluency_PhiEval_scores.extend(utterance['fluency_PhiEval'])
        coherency_PhiEval_scores.extend(utterance['coherency_PhiEval'])

# Calculate correlation matrices
coherency_data = [
    coherency_scores,
    coherency_score_values,
    coherency_MistralEval_scores,
    coherency_GPTEval_scores,
    coherency_GPT4Eval_scores,
    coherency_LlamaEval_scores,
    coherency_PhiEval_scores
]
fluency_data = [
    fluency_scores,
    perplexity_values,
    fluency_MistralEval_scores,
    fluency_GPTEval_scores,
    fluency_GPT4Eval_scores,
    fluency_LlamaEval_scores,
    fluency_PhiEval_scores
]

# Compute Pearson correlation matrix
coherency_corr_matrix = np.corrcoef(coherency_data)
fluency_corr_matrix = np.corrcoef(fluency_data)

# Create DataFrames for better visualization
coherency_list_names = [
    'Human',
    'RoBerta',
    'Mistral',
    'GPT3.5',
    'GPT4',
    'Llama',
    'Phi'
]
fluency_list_names = [
    'Human',
    'Pplxty',
    'Mistral',
    'GPT4',
    'GPT3.5',
    'Llama',
    'Phi'
]

coherency_corr_df = pd.DataFrame(coherency_corr_matrix, index=coherency_list_names, columns=coherency_list_names)
fluency_corr_df = pd.DataFrame(fluency_corr_matrix, index=fluency_list_names, columns=fluency_list_names)

print("-----------Coherency Pearson Correlation-------------")
print(coherency_corr_df)

print("-----------Fluency Pearson Correlation-------------")
print(fluency_corr_df)

# Calculate Spearman and Kendall Tau correlations
def calculate_spearman_kendall(data, names):
    spearman_corr = np.zeros((len(data), len(data)))
    kendall_corr = np.zeros((len(data), len(data)))

    for i in range(len(data)):
        for j in range(len(data)):
            spearman_corr[i, j], _ = spearmanr(data[i], data[j])
            kendall_corr[i, j], _ = kendalltau(data[i], data[j])

    spearman_df = pd.DataFrame(spearman_corr, index=names, columns=names)
    kendall_df = pd.DataFrame(kendall_corr, index=names, columns=names)

    return spearman_df, kendall_df

coherency_spearman_df, coherency_kendall_df = calculate_spearman_kendall(coherency_data, coherency_list_names)
fluency_spearman_df, fluency_kendall_df = calculate_spearman_kendall(fluency_data, fluency_list_names)

print("-----------Coherency Spearman Correlation-------------")
print(coherency_spearman_df)

print("-----------Coherency Kendall Tau Correlation-------------")
print(coherency_kendall_df)

print("-----------Fluency Spearman Correlation-------------")
print(fluency_spearman_df)

print("-----------Fluency Kendall Tau Correlation-------------")
print(fluency_kendall_df)

# Perform T-tests
print("----------Coherency T-tests-------------------")
t_stat, _ = ttest_ind(coherency_scores, coherency_GPTEval_scores)
print("GPT T-statistic:", t_stat)
t_stat, _ = ttest_ind(coherency_scores, coherency_GPT4Eval_scores)
print("GPT T-statistic:", t_stat)
t_stat, _ = ttest_ind(coherency_scores, coherency_LlamaEval_scores)
print("Llama T-statistic:", t_stat)
t_stat, _ = ttest_ind(coherency_scores, coherency_MistralEval_scores)
print("Mistral T-statistic:", t_stat)
t_stat, _ = ttest_ind(coherency_scores, coherency_PhiEval_scores)
print("Phi T-statistic:", t_stat)

print("----------Fluency T-tests-------------------")
t_stat, _ = ttest_ind(fluency_scores, fluency_GPTEval_scores)
print("GPT T-statistic:", t_stat)
t_stat, _ = ttest_ind(fluency_scores, fluency_GPT4Eval_scores)
print("GPT4 T-statistic:", t_stat)
t_stat, _ = ttest_ind(fluency_scores, fluency_LlamaEval_scores)
print("Llama T-statistic:", t_stat)
t_stat, _ = ttest_ind(fluency_scores, fluency_MistralEval_scores)
print("Mistral T-statistic:", t_stat)
t_stat, _ = ttest_ind(fluency_scores, fluency_PhiEval_scores)
print("Phi T-statistic:", t_stat)

print("----------Coherency-GPT4 T-tests-------------------")
t_stat, _ = ttest_ind(coherency_GPT4Eval_scores, coherency_GPTEval_scores)
print("GPT T-statistic:", t_stat)
t_stat, _ = ttest_ind(coherency_GPT4Eval_scores, coherency_LlamaEval_scores)
print("Llama T-statistic:", t_stat)
t_stat, _ = ttest_ind(coherency_GPT4Eval_scores, coherency_MistralEval_scores)
print("Mistral T-statistic:", t_stat)
t_stat, _ = ttest_ind(coherency_GPT4Eval_scores, coherency_PhiEval_scores)
print("Phi T-statistic:", t_stat)

print("----------Fluency-GPT4 T-tests-------------------")
t_stat, _ = ttest_ind(fluency_GPT4Eval_scores,fluency_GPTEval_scores)
print("GPT T-statistic:", t_stat)
t_stat, _ = ttest_ind(fluency_GPT4Eval_scores, fluency_LlamaEval_scores)
print("Llama T-statistic:", t_stat)
t_stat, _ = ttest_ind(fluency_GPT4Eval_scores, fluency_MistralEval_scores)
print("Mistral T-statistic:", t_stat)
t_stat, _ = ttest_ind(fluency_GPT4Eval_scores, fluency_PhiEval_scores)
print("Phi T-statistic:", t_stat)
