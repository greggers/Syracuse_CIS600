from bayes_theory import bayes_theorem

class BayesianNetwork:
    def __init__(self):
        # Define the conditional probability tables
        self.p_rain = {
            'T': 0.2,
            'F': 0.8
        }
        
        self.p_sprinkler_given_rain = {
            ('T', 'T'): 0.01,
            ('T', 'F'): 0.4,
            ('F', 'T'): 0.99,
            ('F', 'F'): 0.6
        }
        
        self.p_grass_wet_given_sprinkler_rain = {
            ('T', 'T', 'T'): 0.99,
            ('F', 'T', 'T'): 0.01,
            ('T', 'F', 'T'): 0.8,
            ('F', 'F', 'T'): 0.2,
            ('T', 'T', 'F'): 0.9,
            ('F', 'T', 'F'): 0.1,
            ('T', 'F', 'F'): 0.01,
            ('F', 'F', 'F'): 0.99
        }
    
    def joint_probability(self, rain, sprinkler, grass_wet):
        """Calculate the joint probability P(Rain, Sprinkler, Grass_Wet)"""
        p_r = self.p_rain[rain]
        p_s_given_r = self.p_sprinkler_given_rain[(sprinkler, rain)]
        p_g_given_s_r = self.p_grass_wet_given_sprinkler_rain[(grass_wet, sprinkler, rain)]
        
        return p_r * p_s_given_r * p_g_given_s_r
    
    def probability_grass_wet(self):
        """Calculate P(Grass_Wet = True)"""
        # Sum up all joint probabilities where Grass_Wet = True
        prob_grass_wet = 0.0
        
        for rain in ['T', 'F']:
            for sprinkler in ['T', 'F']:
                prob_grass_wet += self.joint_probability(rain, sprinkler, 'T')
        
        return prob_grass_wet
    
    def probability_grass_wet_given_rain(self, rain):
        """Calculate P(Grass_Wet=True | Rain=True)"""
        # Calculate P(Grass_Wet=True | Rain=True)
        prob_grass_wet_given_rain = 0.0

        for sprinkler in ['T', 'F']:
            prob_grass_wet_given_rain += self.joint_probability(rain, sprinkler, 'T')

        return prob_grass_wet_given_rain
    
    def probability_rain_given_grass_wet(self):
        """Calculate P(Rain=T | Grass_Wet=T) using Bayes' theorem"""
        
        # Get P(Rain=T) - this is our prior probability
        prior_prob = self.p_rain['T']
        
        # Calculate P(Grass_Wet=T | Rain=T) - this is our likelihood
        likelihood = self.probability_grass_wet_given_rain('T')
        
        # Calculate P(Grass_Wet=T | Rain=F) - this is our false positive rate
        false_positive_rate = self.probability_grass_wet_given_rain('F')
        
        # Apply Bayes' theorem using the imported function
        p_rain_true_given_grass_wet = bayes_theorem(
            prior_prob=prior_prob,
            likelihood=likelihood,
            false_positive_rate=false_positive_rate
        )
        
        return p_rain_true_given_grass_wet


def main():
    bn = BayesianNetwork()
    
    # Calculate the probability that the grass is wet
    prob_grass_wet = bn.probability_grass_wet()

    # Calculate the probability that the grass is wet given it rained
    prob_grass_wet_given_rain_T = bn.probability_grass_wet_given_rain('T')
    prob_grass_wet_given_rain_F = bn.probability_grass_wet_given_rain('F')

    # Calculate the probability that it rained given the grass is wet
    prob_rain_given_grass_wet = bn.probability_rain_given_grass_wet()
    
    print("Sprinkler Bayesian Network Problem:")
    print("P(RAIN=T) = 0.2")
    print("P(RAIN=F) = 0.8")
    print("P(SPRINKLER=T| RAIN=F) = 0.4")
    print("P(SPRINKLER=F| RAIN=F) = 0.6")
    print("P(SPRINKLER=F| RAIN=T) = 0.99")
    print("P(SPRINKLER=T| RAIN=T) = 0.01")
    print("P(GRASS_WET=T | SPRINKER=F, RAIN=F) = 0.01")
    print("P(GRASS_WET=F | SPRINKER=F, RAIN=F) = 0.99") 
    print("P(GRASS_WET=T | SPRINKER=F, RAIN=T) = 0.8")
    print("P(GRASS_WET=F | SPRINKER=F, RAIN=T) = 0.2")
    print("P(GRASS_WET=T | SPRINKER=T, RAIN=F) = 0.8")
    print("P(GRASS_WET=F | SPRINKER=T, RAIN=F) = 0.2")
    print("P(GRASS_WET=T | SPRINKER=T, RAIN=T) = 0.99")
    print("P(GRASS_WET=F | SPRINKER=T, RAIN=T) = 0.01")
    print(f"Probability that the grass is wet: {prob_grass_wet:.4f} or {prob_grass_wet:.2%}")
    print(f"Probability that the grass is wet given it rained: {prob_grass_wet_given_rain_T:.4f} or {prob_grass_wet_given_rain_T:.2%}")
    print(f"Probability that the grass is wet given it didn't rain: {prob_grass_wet_given_rain_F:.4f} or {prob_grass_wet_given_rain_F:.2%}")
    print(f"Probability that it rained given the grass is wet: {prob_rain_given_grass_wet:.4f} or {prob_rain_given_grass_wet:.2%}")

if __name__ == "__main__":
    main()