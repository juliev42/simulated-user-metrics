import pandas as pd
import numpy as np
import names
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

## funct and reference library
def get_date_n_weeks_ago(n):
    return (pd.Timestamp.now() - pd.DateOffset(weeks=n)).strftime("%m_%d_%Y")

#the following 3 could also be a dict, but laying out for clarity 
departments = ['Marketing', 'Sales', 'Engineering', 'Product', 'Design', 'Customer Success', 'Finance', 'People', 'Legal', 'Other']
department_assignment_probs = [0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05] #sum to 1
department_prob_active = [0.8, 0.5, 0.6, 0.8, 0.9, 0.9, 0.3, 0.5, 0.4, 0.2] # probability someone in a particular dept is active on a given day
messages_per_day = 3 #messages if active

acct_created = pd.Timestamp(2019, 1, 1)
names_list = [names.get_full_name() for i in range(100)]

def make_init_data_frame(num_employees):
    data_dict = {}
    for i in range(num_employees):
        person = names.get_full_name()
        department_ind = np.random.choice(len(departments), p=department_assignment_probs)
        department = departments[department_ind]
        days_active = np.random.binomial(5, department_prob_active[department_ind]) #modeling days active as a binomial distribution
        messages_sent = np.random.poisson(days_active*messages_per_day) #modeling messages sent as a poisson distribution with lambda = days_active*messages_per_day
        creation_date = acct_created
        data_dict[person] = [department, days_active, messages_sent, creation_date]
    df = pd.DataFrame.from_dict(data_dict, orient='index', columns=['department', 'days_active', 'messages_sent', 'acct_creation_date'])
    return df


def get_prob_activity_given_week(week_num):
    #add slack disengagement in first half of departments with time passage
    #add slack engagement in 2nd half of departments with time passage
    # return prob_active for a given week
    # this is unrealistic but just for fun
    prob_active_week = []
    for i in range(len(department_prob_active)):
        if i < len(department_prob_active)/2:
            new_active  = max(0,department_prob_active[i] - week_num*abs(np.random.normal(0, 0.1)))
            prob_active_week.append(new_active)
        else:
            new_active = min(1, department_prob_active[i] + week_num*abs(np.random.normal(0, 0.1)))
            prob_active_week.append(new_active)
    return prob_active_week


def make_longform_data_frame(num_employees, num_weeks):
    data_dict = {}
    for i in range(num_employees):
        person = names.get_full_name()
        department_ind = np.random.choice(len(departments), p=department_assignment_probs)
        department = departments[department_ind]
        week_k_days_active = []
        week_k_messages_sent = []
        creation_date = acct_created
        for k in range(num_weeks):
            prob_active_week_k = get_prob_activity_given_week(k)
            days_active = np.random.binomial(5, prob_active_week_k[department_ind]) #modeling days active as a binomial distribution
            messages_sent = np.random.poisson(days_active*messages_per_day) #modeling messages sent as a poisson distribution with lambda = days_active*messages_per_day
            week_k_days_active.append(days_active)
            week_k_messages_sent.append(messages_sent)
            
        data_dict[person] = [department, creation_date, week_k_days_active, week_k_messages_sent, ]
    df = pd.DataFrame.from_dict(data_dict, orient='index', columns=['department', 'acct_creation_date', 'days_active', 'messages_sent'])
    return df

        
    

st.write('Let\'s simulate some fake engagement data for a Slack workspace. We\'ll start by creating some fake employees.')

num_employees = st.slider('How many employees?', 1, 100, 10)

st.write('How many weeks of data would you like to simulate?')
num_weeks = st.slider('How many weeks?', 1, 20, 1)

st.write('Here are your fake employees:')
df_long = make_longform_data_frame(num_employees, num_weeks)
st.write(df_long[['department',  'acct_creation_date']])

st.write('Here are the data on messages sent for the past {} weeks:'.format(num_weeks))

week_labels = [get_date_n_weeks_ago(i) for i in range(num_weeks)]
longform_messages_sent = pd.DataFrame(df_long['messages_sent'].tolist(), index=df_long.index, columns=[f'm_{w}' for w in week_labels.reverse()])
joined_df = pd.concat([df_long[['department']], longform_messages_sent], axis = 1)
st.write(joined_df)


st.write('Here is a plot of the messages sent for the past {} weeks by department.'.format(num_weeks))

aggregate = joined_df.groupby('department').agg('mean', numeric_only=True)
fig = plt.figure(figsize=(10, 4))
sns.lineplot(data = aggregate.T)
plt.xticks(rotation=45)
plt.ylabel('Messages Sent (per capita)')
plt.xlabel('Week')
plt.title('Avg Messages Sent by Department')

plt.legend(loc=(1.04, 0))

st.pyplot(fig)

st.write('In this simulation, the departments trend towards either maximum engagement or disengagement over time. Design, Engineering, Marketing, Product, and Sales trend towards 0 engagement.')












