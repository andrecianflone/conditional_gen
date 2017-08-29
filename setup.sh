
# Requires the following repo, which is code for
# Learning to Generate Reviews and Discovering Sentiment
# ref: https://arxiv.org/abs/1704.01444
git clone https://github.com/openai/generating-reviews-discovering-sentiment.git sentiment_neuron

# Extract hidden units from openAi model, assumes 'data/all_pdtb.json'
python helper.py
