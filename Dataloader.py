import json
import pandas as pd
import numpy as np

class DataLoader():

    def __init__(self):
        self.load_data();

    def buildClear(self, leaf, rows, clears):
        for row in rows:
            if row['node'] == leaf:
                clears.append(row)

    def addEvent(self, events, event):
        events.append(event)
    
    def load_data(self):
        config_file = json.load(open('configuration.json'))
        dataset = config_file['dataset']['list']
        ground_file = 'GroundTruth/' + dataset + '.txt'
        events = []
        clears = []

        df = pd.read_csv(ground_file, sep=',')
        for row in df.iterrows():
            eventRecord = {
                'name': row[0],
                'node': row[1]['Node'],
                'type': row[1]['Type'],
                'startTime': row[1]['Start'],
                'endTime': row[1]['End']
            }
            self.addEvent(events, eventRecord)

        print(events[0]['node'])
        self.buildClear('leaf1', events, clears)

        starttime = []
        endtime = []
        for data in clears:
            starttime.append(data['startTime'])
            endtime.append(data['endTime'])

        # Load csv files
        print("loading csv files....")
        df_csv = pd.read_csv('Data/DatasetByNodes/leaf1bgpclear_apptraffic_2hourRun.csv',
            low_memory=False).dropna().drop('Unnamed: 0', axis=1)
        print("done!")

        df_csv['result'] = False

        # print(df_csv.result)

        for i in range(len(starttime)):
            print(starttime[i], endtime[i])
            df_csv.loc[(df_csv.time >= starttime[i]) & (df_csv.time <= endtime[i]),'result']=True
            # df_csv.loc[(df_csv.time >= starttime[i] & df_csv.time <= endtime[i]), 'result'] = True
        self.data = df_csv
    
    def get_dataset(self):
        return self.data

if __name__ == "__main__":
    dataloader = DataLoader()
    print(dataloader.get_dataset())
