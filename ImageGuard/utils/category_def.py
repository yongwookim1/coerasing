class Categories:
    def __init__(self):
        self.dim = {}
        categories = [
            ['Male', 'Female', 'Children', 'Teenager', 'Young adult', 'Middle-aged', 'Elderly', 'Caucasian', 'Indian', 'Asian', 'African', 'Latino'], 
            ['Sexual', 'Hate', 'Humiliation', 'Violence', 'Illegal activity', 'Disturbing'],
            ['Public figures', 'Personal identification documents', 'Intellectual property violation'],
        ]
        self.cat2dim = {}
        self.cat_and_dim = []
        self.dim['Fairness'] = categories[0]
        self.dim['Toxicity'] = categories[1] 
        self.dim['Privacy'] = categories[2]
        self.dim['Safe'] = ['safe']

        self.unsafe_categories = [x.lower() for x in categories[1] + categories[2]]
        keylist = list(self.dim.keys())
        
        for key in keylist:
            for idx, one in enumerate(self.dim[key]):
                self.cat2dim[one.lower()] = key
                self.cat_and_dim.append([one, key])
                self.dim[key][idx] = one.lower()

        self.cat2dim['safe'] = 'safe'
        self.cat2dim['unsafe'] = 'unsafe'
        self.cat_and_dim.append(['safe', 'safe'])

        self.all = [y.lower() for x in categories for y in x]
        self.all.append('safe')
        self.all.append('unsafe')