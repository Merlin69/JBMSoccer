import csv
import datetime
import ToolBox
import random


class Data:
    def __init__(self, filename, nb_times):
        self.filename = filename
        self.country_to_id = {}
        self.id_to_country = []
        self.nb_teams = 0
        self.nb_times = nb_times
        max_time = 0
        min_time = datetime.date(2100, 1, 1).toordinal()
        with open(filename, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                if row:
                    for s in row[:2]:
                        s_cropped = ToolBox.format_name(s)
                        if s_cropped not in self.country_to_id:
                            self.country_to_id[s_cropped] = self.nb_teams
                            self.id_to_country.append(s_cropped)
                            self.nb_teams += 1
                    time = ToolBox.date_to_number(row[4])
                    if time > max_time:
                        max_time = time
                    if time < min_time:
                        min_time = time
        self.date_to_id = lambda x: ToolBox.gen_date_to_id(x, self.nb_times, max_time,min_time)
        self.train = {}
        self.test = {}
        self.elo = None
        self.metadata = {'nb_teams': self.nb_teams, 'nb_times': self.nb_times}

    def get(self, p):
        self.train = dict(dict(time=[], team_h=[], team_a=[], res=[], score_h=[], score_a=[]), **self.metadata)
        self.test = dict(dict(time=[], team_h=[], team_a=[], res=[], score_h=[], score_a=[]), **self.metadata)
        with open(self.filename, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                if row:
                    [team1, team2, score1, score2, time] = row[:5]
                    date = self.date_to_id(time)
                    if random.random() < float(p):
                        proxy = self.train
                    else:
                        proxy = self.test
                    self.append_match(proxy, team1, team2, score1, score2, date)

    def create_matches(self, matches):
        formated_matches = dict(dict(time=[], team_h=[], team_a=[], res=[], score_h=[], score_a=[]), **self.metadata)
        for team_h, team_a, date in matches:
            if date == 'last':
                date = self.nb_times - 1
            self.append_match(formated_matches, team_h, team_a, 0, 0, date)
        return formated_matches

    def append_match(self, proxy, team1, team2, score1, score2, date):
        proxy['time'].append(ToolBox.make_vector(date, self.nb_times))
        id_team_h = self.country_to_id[ToolBox.format_name(team1)]
        id_team_a = self.country_to_id[ToolBox.format_name(team2)]
        proxy['team_h'].append(ToolBox.make_vector(id_team_h, self.nb_teams))
        proxy['team_a'].append(ToolBox.make_vector(id_team_a, self.nb_teams))
        score_team_h = min(int(score1), 9)
        score_team_a = min(int(score2), 9)
        proxy['score_h'].append(ToolBox.make_vector(score_team_h, 10))
        proxy['score_a'].append(ToolBox.make_vector(score_team_a, 10))
        proxy['res'].append(ToolBox.result(int(score1) - int(score2)))

    def set_elos(self, elo):
        self.elo = elo

    def get_elos(self, countries=None, times='all'):
        if countries is None:
            countries = [self.id_to_country[i] for i in range(self.nb_teams)]
        else:
            countries = map(ToolBox.format_name, countries)
        if times == 'all':
            times = range(self.nb_times)
        elif times == 'last':
            times = [self.nb_times - 1]
        elos = {}
        for country in countries:
            elos[country] = [200*self.elo[self.country_to_id[country]][t] for t in times]
        return elos
