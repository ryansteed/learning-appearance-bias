from appearance_bias.api import interpret

# Random Face Validation
interpret(
	'Trustworthy', 
	['data/maxdistinct_aligned'], 
	'data/random_aligned', 
	file='png',
	ground_truth=False,
	save=True,
	num_samples=5000
)

# Politician Bias
interpret(
    'Competent', 
    ['data/maxdistinct_aligned'], 
    '../data/politicians-database_aligned',
    file='png',
    ground_truth=False,
    save=True,
    num_samples=5000
)