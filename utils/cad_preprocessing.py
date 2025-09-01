import pandas as pd

path = 'data/raw/cad/cad_v1_1.tsv'
cad_raw_df = pd.read_csv(path, sep='\t')

# keep only rows with annotation_Target != "NA"
cad_df = cad_raw_df[cad_raw_df['annotation_Target'] != 'NA']




# eliminate rows of which occurrences are less than 5
cad_df = cad_df[cad_df.groupby('annotation_Target')['annotation_Target'].transform('size') >= 3]


# write the value counts to a file
cad_df["annotation_Target"].value_counts().to_csv('data/raw/cad/cad_target_counts.tsv', sep='\t')

cad_df = cad_df[cad_df['meta_text'].str.len() >= 7]

#map "gay men", "gay people", "sexual and gender minorities" all in the same category called "sexuality"
cad_df['annotation_Target'] = cad_df['annotation_Target'].replace({
    # Sexuality and Gender Identity
    "gay men": "sexuality",
    "gay people": "sexuality", 
    "gay women": "sexuality",
    "sexual and gender minorities": "sexuality",
    "transgender people": "sexuality",
    "non-gender dysphoric transgender people": "sexuality",
    "transgender women": "sexuality",
    "lgbtqa community": "sexuality",
    "asexual people": "sexuality",
    
    # Disabilities and Mental Health
    "people with mental disabilities": "disabled",
    "people with autism": "disabled",
    "people with aspergers": "disabled", 
    "people with mental health issues": "disabled",
    "people with disabilities": "disabled",
    "people with down's syndrome": "disabled",
    "people with cerebral palsy": "disabled",
    "people with physical disabilities": "disabled",
    "people with dwarfism": "disabled",
    "people with drug problems": "disabled",
    
    # Political - Left Wing
    "left-wing people": "left-wing",
    "left-wing people (social justice)": "left-wing",
    "left-wing people (far left)": "left-wing",
    "liberals": "left-wing",
    "democrats": "left-wing",
    "communists": "left-wing",
    "socialists": "left-wing",
    "anarchists": "left-wing",
    
    # Political - Right Wing  
    "right-wing people": "right-wing",
    "right-wing people (alt-right)": "right-wing",
    "right-wing people (far right)": "right-wing",
    "conservatives": "right-wing",
    "republicans": "right-wing",
    "donald trump supporters": "right-wing",
    "nazis": "right-wing",
    "white supremacists": "right-wing",
    "nationalists": "right-wing",
    
    # Political - Center/Other
    "centrists": "DROP",
    "libertarians": "DROP",
    "monarchists": "DROP",
    "politicians": "DROP",
    
    # Political Ideologies
    "zionists": "DROP",
    "capitalists": "DROP",
    
    # Race/Ethnicity - Black
    "black people": "black",
    "black men": "black", 
    "black women": "black",
    "people of color": "black",
    "people from africa": "black",
    
    # Race/Ethnicity - White
    "white people": "white",
    "white men": "white",
    "white women": "white",
    "people from america": "white",
    "people from britain": "white",
    "people from europe": "white",
    "people from ireland": "white",

    # Race/Ethnicity - Asian  
    "chinese women": "asian",
    "chinese men": "asian",
    "asian men": "asian", 
    "asian women": "asian",
    "people from china": "asian",
    "people from asia": "asian",
    
    # Race/Ethnicity - Other
    "non-white people": "DROP", 
    "ethnic minorities": "DROP",
    "mixed race/ethnicity": "DROP",
    "romani people": "DROP",
    "arabs": "DROP",
    "latinx": "latinx",
    "people from mexico": "latinx",
    "indigenous people": "DROP",
    
    # Religious Groups
    "muslims": "muslims",
    "muslim women": "muslims",
    "jewish people": "jewish",
    "christians": "christians", 
    "catholics": "christians",
    "hindus": "hindus",
    "atheists": "DROP",
    "priests": "DROP",
    
    # Geographic/Immigration
    "people from pakistan": "DROP",
    "people from india": "DROP",
    "people from israel": "DROP",
    "people from south africa": "DROP",

    "illegal immigrants": "immigrants",
    "immigrants": "immigrants",
    "refugees": "immigrants",
    "foreigners": "immigrants",
    
    "non-masculine men": "DROP",
    
    # Feminism/Gender Rights
    "feminists": "feminists",
    "feminists (male)": "feminists",
    "feminists (trans-exclusionary radical)": "feminists", 
    "feminists (radical)": "feminists",
    "men's rights activists": "feminists",
    
    # Age/Class
    "elderly people": "age",
    "young people": "age", 
    "working class people": "DROP",
    "rich people": "DROP",
    "poor people": "DROP",
    
    # Subcultures/Communities
    "involuntary celibates": "DROP",
    "gamers": "DROP",
    'fans of anthropomorphic animals ("furries")': "DROP",
    "fans of japanese culture (western)": "DROP",
    "vegans/vegetarians": "DROP",
    
    # Activists
    "activists (anti-fascist)": "activists",
    "activists (animal rights)": "activists",
    "activists (anti-vaccination)": "activists", 
    "activists (black rights)": "activists",
    "activists (anti-abortion)": "activists",
    
    # Professions/Roles
    "moderators": "DROP",
    "police officers": "DROP",
    "teachers": "DROP",
    "soldiers": "DROP", 
    "journalists": "DROP",
    "sex workers": "DROP",
    "baristas": "DROP",
    "office workers - charities": "DROP",
    
    # Historical/Specific Individuals
    "brenton tarrant": "DROP",
    "hitler": "DROP",
    
    # Historical Events
    "holocaust": "DROP",
    "victims of sandy hook shooting": "DROP", 
    "cromwellian conquest of ireland": "DROP",
    
    # Other
    "cultists": "DROP"
})

# drop rows with "DROP" in annotation_Target
cad_df = cad_df[cad_df['annotation_Target'] != 'DROP']


print(cad_df["annotation_Target"].value_counts())


# total number of occurrencies:
print(f"num occurrencies: {cad_df.shape[0]}")

df = cad_df[["id", "meta_text", "annotation_Target"]]
df = df.rename(columns={
            'id': 'ID',
            'meta_text': 'text',
            'annotation_Target': 'target'
        })

df["text"] = df["text"].apply(lambda x: x.replace("[linebreak]", ""))

# remove multiple subsequenst spaces in the text
df["text"] = df["text"].str.replace(r'\s+', ' ', regex=True)

# remove rows where text is longer than 1500 chars
print(f"num of strings longer than 1500 chars: {df[df['text'].str.len() > 1500].shape[0]}")
df = df[df["text"].str.len() <= 1500]


# save the processed dataframe
df.to_csv('data/processed/cad/cad_v1_1_processed.tsv', sep='\t', index=False)