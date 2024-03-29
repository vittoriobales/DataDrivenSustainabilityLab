# -*- coding: utf-8 -*-
"""
@author: vbalestrieri

"""

# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from torch.utils.data import DataLoader
import torch
from torch.nn import Dropout
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

##################### DATA LOADING ###########################

# Load datasets
Emission_factors = pd.read_excel(#insert path) #contains BEA Code	USEEIO Good & Service Name with the associated kg C02 x $
dataset = pd.read_excel(#insert path) #contains series of exemples generated
dataset['Code'] = dataset['Code'].astype(str)
df_cleaned = dataset


label_encoder = LabelEncoder()
df_cleaned['Encoded_Code'] = label_encoder.fit_transform(df_cleaned['Code'].astype(str))


# Stratified Splitting
X = df_cleaned['Description']  # Features
y = df_cleaned['Encoded_Code']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) #stratify assuring a global rappresentatio of all the BEA Codes
train_data = pd.concat([X_train, y_train], axis=1) 
test_data = pd.concat([X_test, y_test], axis=1)



####################### MODEL TRAINING ############################# 
# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base') #using pretrained Roberta LLM model

def encode_data(df): #convert the textual descriptions into a format suitable for the model
    inputs = tokenizer(df['Description'].tolist(), padding=True, truncation=True, return_tensors="pt")
    # Convert labels to Long type
    labels = torch.tensor(df['Encoded_Code'].values, dtype=torch.long)
    dataset = [{'input_ids': inputs['input_ids'][i], 'attention_mask': inputs['attention_mask'][i], 'labels': labels[i]} for i in range(len(df))]
    return dataset


train_dataset = encode_data(train_data)
val_dataset = encode_data(test_data)

# Model with dropout for regularization
class CustomRoberta(RobertaForSequenceClassification):
    def __init__(self, config):
        super(CustomRoberta, self).__init__(config)
        self.dropout = Dropout(0.3)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # Apply dropout to the pooled output (not to the raw model outputs)
        pooled_output = self.dropout(outputs[1])
        return (outputs[0], pooled_output) if labels is not None else pooled_output


model = CustomRoberta.from_pretrained('roberta-base', num_labels=len(y_train.unique()))


class MyEarlyStoppingCallback(EarlyStoppingCallback):
    def __init__(self, early_stopping_patience=3, early_stopping_threshold=0.0):
        super().__init__()
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        
        
# Initialize the early stopping callback
early_stopping_callback = MyEarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)


# Training with the early stopping callback
training_args = TrainingArguments(
    output_dir="./fine-tuned-model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    save_strategy="epoch",  # Set save strategy to epoch
    evaluation_strategy="epoch"
)

# MODEL TRAINING - FINE TUNING !
 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[early_stopping_callback]  # Add the early stopping callback here
)

trainer.train()
################################### TESTING AND PREDICTIONS  #########################

#lOAD already fine-tuned model
model = RobertaForSequenceClassification.from_pretrained(r"C:\AppNus\fine-tuned-model\checkpoint-10256")


#for testing 
remaining_ledger_entries = test_data['Description'] 
tokenizer = RobertaTokenizer.from_pretrained('file:C:\AppNus\fine-tuned-model\checkpoint-10256')
remaining_encodings = tokenizer(remaining_ledger_entries.tolist(), truncation=False, padding=True, return_tensors='pt')


##### EVALUATION #######
# Evaluate the model
val_loader = DataLoader(val_dataset, batch_size=8)

model.eval()
true_labels = []
predictions = []

for batch in val_loader:
    input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
    attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
    labels = batch['labels'].to('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    batch_predictions = torch.argmax(logits, dim=1)
    predictions.extend(batch_predictions.tolist())
    true_labels.extend(labels.tolist())


# Convert numerical labels back to original labels if necessary
# original_labels = label_encoder.inverse_transform(true_labels)

# Calculate metrics
accuracy = sum([1 for true, pred in zip(true_labels, predictions) if true == pred]) / len(true_labels)
precision = precision_score(true_labels, predictions, average='macro')
recall = recall_score(true_labels, predictions, average='macro')
f1 = f1_score(true_labels, predictions, average='macro')

print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")



decoded_predictions = label_encoder.inverse_transform(predictions)
decoded_true_labels = label_encoder.inverse_transform(true_labels)

# Now WE can use `decoded_predictions` and `decoded_true_labels` for a more interpretable comparison.



emission_factors_dict = {str(key).strip(): value for key, value in zip(Emission_factors['BEA Code'], Emission_factors['KGCO2eq/ $'])}


# Compute emissions for each purchase line item
remaining_ledger_entries = test_data['Description']
remaining_ledger_entries_emission = []



for entry, predicted_commodity in zip(remaining_ledger_entries, decoded_predictions):
    # Debugging: Check if the predicted commodity is in the emission factors dict
    if predicted_commodity in emission_factors_dict:
        emission_factor = emission_factors_dict[predicted_commodity]
        print(f"Found emission factor for {predicted_commodity}: {emission_factor}")
    else:
        print(f"Emission factor for {predicted_commodity} not found.")
        emission_factor = 0.0  # Default to 0 if not found

    # Find the expense amount for the corresponding entry
    filtered_rows = dataset[dataset['Description'] == entry]
    if not filtered_rows.empty:
        normalized_expense = filtered_rows['Amount'].values[0]
        print(f"Normalized expense for '{entry}': {normalized_expense}")
    else:
        print(f"No matching entry found in dataset for description: {entry}")
        normalized_expense = 0  # Default to 0 if not found

    # Calculate emissions
    emissions = normalized_expense * emission_factor
    print(f"Emissions for '{entry}': {emissions}")  # Debugging: Print calculated emissions

    remaining_ledger_entries_emission.append({
        "Description": entry,
        "Predicted_Commodity": predicted_commodity,
        "Emission_Factor": emission_factor,
        "Emissions": emissions
    })


# Create a DataFrame for the computed emissions
emission_dataset = pd.DataFrame(remaining_ledger_entries_emission)

# Display the results
print(emission_dataset[['Description', 'Predicted_Commodity', 'Emission_Factor', 'Emissions']])
