import pickle
import pandas as pd


#Writing pickle file

# first_run = pd.read_csv(r"X-Ray\Data\02-02-2023\MEGA DATA.csv",skiprows=0)
# pickle.dump(first_run, open(r'X-Ray\Data\02-02-2023\MEGA DATA.pkl', 'wb'))


#reading pickle
data = pickle.load(open(r'X-Ray\Data\02-02-2023\MEGA DATA.pkl', 'rb'))
#gets name of columns
print(data.keys())
column_names = ['E_1 / keV', 'Mo plate cal', 'Cu straight through calibrate',
       'Unknown - Tungsten (W)', 'Unknown - Ti', 'Pb', ' In +  plastic',
       'Tin/Copper', 'Fe Alloy', 'Iron Zinc', 'Pure Iron',
       'Gold (Ganel’s pendant)', 'Iron Nickel', 'Gold (Ganel’s chain)',
       'Nickel (Tulsi’s Ring)', 'inconclusive (Ben’s Ring)',
       'Manganin (Cu Mn Ni)', 'Iron (with possible mixing)',
       'Stainless steel ', 'Solder (tin)', 'Nickel-Brass (Ganel’s Key)',
       '50 Florint ', '10 peso (Copper Zinc)', 'Israel (Copper Zinc Nickel)',
       'HK Dollar', '5 Euro Cent (primarily copper)', 'Indium', 'Pure Iron.1',
       'Pure Nickel']
