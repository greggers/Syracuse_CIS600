def bayes_theorem(prior_prob, likelihood, false_positive_rate):
    """
    Calculate posterior probability using Bayes' theorem.
    
    Args:
        prior_prob: Prior probability of the hypothesis (e.g., having disease)
        likelihood: Probability of evidence given hypothesis is true (e.g., test positive given disease)
        false_positive_rate: Probability of evidence given hypothesis is false (e.g., test positive given no disease)
    
    Returns:
        Posterior probability (e.g., probability of disease given positive test)
    """
    # Calculate the marginal probability of the evidence
    marginal = (prior_prob * likelihood) + ((1 - prior_prob) * false_positive_rate)
    
    # Calculate the posterior probability using Bayes' theorem
    posterior = (prior_prob * likelihood) / marginal
    
    return posterior

def main():
    # Given information for the disease test problem
    disease_prevalence = 0.05  # 5% of people have the disease
    test_sensitivity = 0.90    # 90% accurate for detecting disease in people who have it
    test_specificity = 0.80    # 80% accurate for non-detections in people who don't have it
    
    # The false positive rate is 1 - specificity
    false_positive_rate = 1 - test_specificity
    
    # Calculate the probability of having the disease given a positive test
    prob_disease_given_positive = bayes_theorem(
        prior_prob=disease_prevalence,
        likelihood=test_sensitivity,
        false_positive_rate=false_positive_rate
    )
    
    print("Disease Test Problem:")
    print(f"Disease prevalence: {disease_prevalence:.2%}")
    print(f"Test sensitivity (true positive rate): {test_sensitivity:.2%}")
    print(f"Test specificity (true negative rate): {test_specificity:.2%}")
    print(f"Probability of having the disease given a positive test: {prob_disease_given_positive:.2%}")

if __name__ == "__main__":
    main()