# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

 import numpy as np
 import pandas as pd
 

####################################################################################################

RMData_10 = pd.read_csv("EXTERNAL-Foundations_STAAR_SmarterSolving_events_20170501-20170531.csv", usecols = ['Student ID','School ID','Teacher ID','Grade level','Timestamp','Datetime','Mode','Current objective','Content item','Event type','Content type','Try number','Student answer','Result','Hints','Purchased Item'])


Skill_parameters = pd.read_csv('SkillParameters_Aggregate.csv')


###################################################################################################

def slice_dataset(RMData):
    student_index = RMData['Student ID'].unique()
    RMData = RMData.set_index('Student ID')
    if len(student_index)%2 == 0:
        x = round(len(student_index)/2)
    else:
        x = round((len(student_index))/2)+1
    RMData = RMData.loc[student_index[0]:student_index[x], :]
    return RMData

# Slice dataset
RMData_10_A = slice_dataset(RMData_10)
RMData_10 = RMData_10.set_index("Student ID")
RMData_10_B = RMData_10.loc[~RMData_10.index.isin(RMData_10_A.index)]
RMData_10_A = RMData_10_A.reset_index()
RMData_10_B = RMData_10_B.reset_index()



# Construct final dataset and aggregate dataset
RMData_10_B = FinalRMDataSet(RMData_10_B)
Aggregate_10_B = Aggregate_RMData(RMData_10_B)
Aggregate_10_B.to_csv('Aggregate_10_B.csv', index = False, na_rep = '.')


####################################################################################################

 
 def FinalRMDataSet(RMData):
     BKTData = Construct_BKTset(RMData)
     BKTData = BKT_Calculator(BKTData)
     Concat_BKTData = BKTData.loc[:,['Index_RMData','Ln-1','Ln-1_Result', 'Ln', 'Guess', 'Carelessness']]
     Concat_BKTData = Concat_BKTData.set_index('Index_RMData')
     RMData = pd.concat([RMData, Concat_BKTData], axis = 1)
     ItemCorrectness = Correctness_of_items(BKTData)
     RMData = pd.merge(RMData, ItemCorrectness, on = 'item', how = 'left')
     RMData = Create_features(RMData)
     #RMData = Create_streakcolumn(RMData)
     return RMData

 ###################################################################################################
    
def Create_features(RMData):
    # TimeUsage
    RMData['Hints'] = RMData.Hints.fillna(0)
    student = list(RMData.student)
    Timestamp = list(RMData.Timestamp)
    TimeUsage = [0]*len(RMData)
    for i in iter(range(len(RMData)-1)):
      if student[i] != student[i+1]:
          TimeUsage[i] = 0
      elif student[i] == student[i+1]:
          TimeUsage[i] = Timestamp[i+1] - Timestamp[i]
    RMData["TimeUsage"] = TimeUsage
    # Correctness of speedgame
    RMData['If_speedgame'] = RMData.ContentType.map({'Speedgame':1})
    RMData['Result_of_speedgame'] = np.where((RMData.If_speedgame == 1) &\
          (RMData.Result == 'correct'), 1, np.where((RMData.If_speedgame == 1) &\
           (RMData.Result == 'incorrect'), 0, np.nan))
    # Correctness of theory
    RMData['If_theory'] = RMData.ContentType.map({'Theory':1})
    RMData['Result_of_theory'] = np.where((RMData.If_theory == 1) &\
          (RMData.Result == 'correct'), 1, np.where((RMData.If_theory == 1) &\
           (RMData.Result == 'incorrect'), 0, np.nan))
    # Time spent on Theory(Basic practice)
    RMData['Basic_practice_time'] = np.where(RMData.If_theory == 1, \
            RMData.TimeUsage, 0)
    # Correctness of notestest
    RMData['If_notestest'] = RMData.ContentType.map({'Notestest':1})
    RMData['Result_of_notestest'] = np.where((RMData.If_notestest == 1) &\
          (RMData.Result == 'correct'), 1, np.where((RMData.If_notestest == 1) &\
           (RMData.Result == 'incorrect'), 0, np.nan))
    # Correctness of A-level Problem
    RMData['If_ProblemA'] = RMData.ContentType.map({'Problem A':1})
    RMData['Result_of_ProblemA'] = np.where((RMData.If_ProblemA == 1) &\
          (RMData.Result == 'correct'), 1, np.where((RMData.If_ProblemA == 1) &\
           (RMData.Result == 'incorrect'), 0, np.nan))
    # Correctness of B-level Problem
    RMData['If_ProblemB'] = RMData.ContentType.map({'Problem B':1})
    RMData['Result_of_ProblemB'] = np.where((RMData.If_ProblemB == 1) &\
          (RMData.Result == 'correct'), 1, np.where((RMData.If_ProblemB == 1) &\
           (RMData.Result == 'incorrect'), 0, np.nan))
    # Correctness of C-level Problem
    RMData['If_ProblemC'] = RMData.ContentType.map({'Problem C':1})
    RMData['Result_of_ProblemC'] = np.where((RMData.If_ProblemC == 1) &\
          (RMData.Result == 'correct'), 1, np.where((RMData.If_ProblemC == 1) &\
           (RMData.Result == 'incorrect'), 0, np.nan))
    # Time spent on Wall of Mastery C-level Problem
    RMData['If_WallofMastery'] = RMData.Mode.map({'WALL_OF_MASTERY':1})
    RMData['WallofMastery_ProblemC_time'] = np.where(RMData.If_WallofMastery == 1, \
            RMData.TimeUsage, 0)
    # Whether the student take on RMG
    RMData['If_RMG'] = RMData.Mode.map({'RMG':1})
    # My place and decorating room (time)
    RMData['If_myplace'] = RMData.Mode.map({"MY_PLACE":1})
    RMData['Myplace_time'] = np.where(RMData.If_myplace == 1, \
            RMData.TimeUsage, 0)
    RMData['If_myplace_20s'] = ((RMData.If_myplace == 1) & (RMData.Myplace_time >= 20))\
    .map({True:1, False:0})
    RMData['Myplace_20s_time'] = np.where(RMData.If_myplace_20s == 1, \
          RMData.TimeUsage, 0)
    # Speed in Math Race
    RMData['Speedgame_totaltime'] = np.where(RMData.If_speedgame == 1, \
          RMData.TimeUsage, 0)
    RMData['Speedgame_totalitems'] = np.where(((RMData.If_speedgame == 1)\
          & (RMData.EventType == "Submit")), RMData.StudentAnswer.str.count('='),0)
    # Parse Datetime
    RMData.Datetime = pd.to_datetime(RMData.Datetime, utc = -6)
    # Weekday
    RMData['Weekday'] = RMData.Datetime.dt.weekday_name
    # School begin time
    School_begin = RMData.loc[:,'Datetime'].dt.year.astype(str) +\
    "-" + RMData.loc[:,'Datetime'].dt.month.astype(str) + "-" \
    + RMData.loc[:,'Datetime'].dt.day.astype(str) + " 09:00"
    School_begin = pd.to_datetime(School_begin, utc = -6)
    # School closure time
    School_closure = RMData.loc[:,'Datetime'].dt.year.astype(str) +\
    "-" + RMData.loc[:,'Datetime'].dt.month.astype(str) + "-" \
    + RMData.loc[:,'Datetime'].dt.day.astype(str) + " 15:20"
    School_closure = pd.to_datetime(School_closure, utc = -6)
    # Day begins time
    Day_begin = RMData.loc[:,'Datetime'].dt.year.astype(str) +\
    "-" + RMData.loc[:,'Datetime'].dt.month.astype(str) + "-" \
    + RMData.loc[:,'Datetime'].dt.day.astype(str) + " 00:00:01"
    Day_begin = pd.to_datetime(Day_begin, utc = -6)
    # Day ends time
    Day_end = RMData.loc[:,'Datetime'].dt.year.astype(str) +\
    "-" + RMData.loc[:,'Datetime'].dt.month.astype(str) + "-" \
    + RMData.loc[:,'Datetime'].dt.day.astype(str) + " 23:59:59"
    Day_end = pd.to_datetime(Day_end, utc = -6)
    # Whether at School time
    RMData['If_school_time'] = np.where(((RMData.Datetime > Day_begin) & \
      (RMData.Datetime < School_begin)), 1, \
    np.where(((RMData.Datetime > School_closure) & \
              (RMData.Datetime < Day_end)), 1, 0))
    # If it is weekend
    RMData['Weekend'] = RMData.Weekday.map({'Saturday': 1, 'Sunday': 1})
    # How much time spend on RM at home?
    RMData['Time_at_home'] = np.where((RMData.If_school_time == 1) | (RMData.Weekend == 1),\
          RMData.TimeUsage, 0)
    # Proportion of wrong actions where student take more than N seconds
    RMData['Wrongactions_overthan6s'] = np.where((RMData.Result == "incorrect") &\
          (RMData.TimeUsage > 6), 1, 0)
    # Help avoidance, not requesting help on poorly known skill
    RMData['Nohelp_poorknown'] = np.where((RMData['Ln-1'] <= 0.7) & (RMData.Hints < 1), 1, 0)
    # Not requesting help on well known skill
    RMData['Nohelp_wellknown'] = np.where((RMData['Ln-1'] >= 0.75) & (RMData.Hints < 1), 1, 0)
    # Whether the student did diffcult items
    RMData['Solve_difficult_items'] = np.where(RMData.Item_correctness\
          < (RMData.Item_correctness.mean()-RMData.Item_correctness.std()), 1, 0)
    # How many points the student spend on purchaseing books?
    RMData['Book'] = np.where((RMData.PurchasedItem =="A Bedroom for Mustafa, Part 1 (200)")\
                        |(RMData.PurchasedItem =="A Bedroom for Mustafa, Part 2 (200)")\
                        |(RMData.PurchasedItem =="A Bedroom for Mustafa, Part 3 (200)")\
                        |(RMData.PurchasedItem =="A Bedroom for Mustafa, Part 4 (200)")\
                        |(RMData.PurchasedItem =="A Bedroom for Mustafa, Part 5 (200)")\
                        |(RMData.PurchasedItem =="Island Dreams (1000)")\
                        |(RMData.PurchasedItem =="The Ant and the Grasshopper (300)")\
                        |(RMData.PurchasedItem =="The Dog and the Reflection (300)")\
                        |(RMData.PurchasedItem =="The Fox and the Crow (300)")\
                        |(RMData.PurchasedItem =="The Fox and the Grapes (200)")\
                        |(RMData.PurchasedItem =="The Tortoise and the Hare (300)")\
                        |(RMData.PurchasedItem =="The Wind of Change, Part 1 (400)")\
                        |(RMData.PurchasedItem =="The Wind of Change, Part 2 (400)")\
                        |(RMData.PurchasedItem =="The Wind of Change, Part 3 (400)"), 1, 0)
    RMData['Cost_on_books'] = np.where(RMData.Book == 1, RMData.PurchasedItem.str.extract(r"(\d{3,})"), 0)
    # The average time from getting incorrect to next response. TimeUsage on Event type == 'Problem shown' \
    # is the time which student solve the current problem; TimeUsage on Event type == 'Submit' or Results == 1\
    # stands for the time which students' self-reflection on feedbacks
    RMData['Time_after_incorrect'] = np.where(RMData.Result == "incorrect", RMData.TimeUsage, 0)
    # Pauses after reading hints
    RMData.Hints = RMData.Hints.astype(str)
    Hints = list(RMData.Hints)
    Pauses_after_hints = [0]*len(RMData)
    for i in iter(range(len(RMData)-1)):
      if student[i] != student[i+1]:
          Pauses_after_hints[i] = 0
      elif (student[i] == student[i+1]) & (Hints[i] == '1.0'):
          Pauses_after_hints[i] = TimeUsage[i+1] + TimeUsage[i]
      else:
          Pauses_after_hints[i] = 0
    RMData["Pauses_after_hints"] = Pauses_after_hints
    # Long pauses after reading hints
    RMData['LongP_afterhints'] = np.where(RMData.Pauses_after_hints >= 18, 1, 0)
    # Short pauses after reading hints
    RMData['ShortP_afterhints'] = np.where(RMData.Pauses_after_hints <= 9, 1, 0)
    # Long pauses and correct
    RMData['LongP_afterhints_correct'] = np.where((RMData.Pauses_after_hints >= 18)\
          & (RMData.Result == 'correct'), 1, 0)
    # Short pauses and correct
    RMData['ShortP_afterhints_correct'] = np.where((RMData.Pauses_after_hints <= 9)\
          & (RMData.Result == 'correct'), 1, 0)
    RMData.Cost_on_books = RMData.Cost_on_books.astype(int)
    RMData.Hints = RMData.Hints.astype(float)
    # Whether the student getting a streak
    RollingSet = RMData.loc[:, ['student', 'Datetime', 'Result', 'Hints']]
    RollingSet = RollingSet.loc[((RollingSet['Result'] == 'correct') | (RollingSet['Result'] == 'incorrect')),:]
    #RollingSet.Datetime = pd.to_datetime(RMData.Datetime, utc = -6)
    RollingSet.sort_values(["student","Datetime"], inplace = True)
    RollingSet['Results_WithoutHints'] = np.where((RollingSet.Result == 'incorrect')|(RollingSet.Hints == 1.0), 0, 1)
    RollingSet['Streak'] = RollingSet.Results_WithoutHints.rolling(5).sum()
    Concat_RollingSet = RollingSet.loc[:,'Streak']
    RMData = pd.concat([RMData, Concat_RollingSet], axis = 1)
    # Switch mode after getting a streak
    #student = list(RMData_00.student)#
    Switch_mode = [0]*len(RMData)
    Off_task = [0]*len(RMData)
    Streak_record = list(RMData.loc[:,'Streak'])
    Student_Mode = list(RMData.loc[:,'Mode'])
    #TimeUsage = list(RMData_00.loc[:,'TimeUsage'])#
    for i in iter(range(len(RMData)-1)):
        if student[i] != student[i+1]:
            Switch_mode[i] = np.nan
        elif (student[i] == student[i+1]) & (Streak_record[i] == 5) & (Student_Mode[i] == Student_Mode[i+1]):
            Switch_mode[i] = 0
        elif (student[i] == student[i+1]) & (Streak_record[i] == 5) & (Student_Mode[i] != Student_Mode[i+1]):
            Switch_mode[i] = 1
    # Off task after getting a streak
    for i in iter(range(len(RMData)-1)):
        if student[i] != student[i+1]:
            Off_task[i] = np.nan
        elif (student[i] == student[i+1]) & (Streak_record[i] == 5) & (TimeUsage[i] >= 80):
            Off_task[i] = 1
        elif (student[i] == student[i+1]) & (Streak_record[i] == 5) & (TimeUsage[i] < 80):
            Off_task[i] = 0
    RMData['OffTask_AfterStreak'] = Switch_mode
    RMData['SwitchMode_AfterStreak'] = Off_task
    # Guess and carelessness larger than 0.5
    RMData['Carelessness'] = RMData.Carelessness.fillna(0)
    RMData['Guess'] = RMData.Guess.fillna(0)
    RMData = RMData.loc[RMData.TimeUsage <= 600, :]
    return RMData

####################################################################################################


def Construct_BKTset(RMData):
    # Rename columns
    RMData.rename(columns = {'Student ID': 'student', 'School ID': 'SchoolID', \
                         'Teacher ID': 'TeacherID', 'Grade level': 'GradeLevel','Current objective':'skill', 'Content item':'item', 'Event type':'EventType','Content type':'ContentType','Try number':'TryNumber','Student answer':'StudentAnswer', 'Purchased Item': 'PurchasedItem'}, inplace = True)
    # Create Results variable
    RMData['Results'] = np.where(RMData['Result'] == 'correct', 1, \
          np.where(RMData['Result'] == 'incorrect', 0, np.nan))
    # Select columns
    BKT = RMData.loc[:, ['student', 'skill', 'item', 'TryNumber', 'Results', 'Hints']]
    # Select rows. Noted: index has been changed!
    newset = BKT.loc[((BKT['Results'] == 1) | (BKT['Results'] == 0)) & (BKT['TryNumber'] == "first"), :]
    # Get rid of the punctation in skill column
    newset.loc[:,'skill'] = newset.loc[:,'skill'].str.replace(r"\s", "")
    r='[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    newset.loc[:,'skill'] = newset.loc[:,'skill'].str.replace(r, "")
    # Add old_index column for future use!
    #newset.loc[:,'Index_relate_to_RMData'] = newset.index
    # Sort
    newset.sort_values(["skill","student"], inplace = True)
    # Reindex
    newset.reset_index(inplace = True)
    newset.rename(columns = {'index':'Index_RMData'}, inplace = True) 
    return newset

####################################################################################################
    
     #The following code is writen for BruceForce-BKT
    '''newset.rename(columns = {'Results': 'right'}, inplace = True)
    newset.loc[:,"lesson"] = "lesson1"
    newset.loc[:,"cell"] = 'cell'
    newset.loc[:,"eol"] = 'eol'
    newset.index.name = "num"
    header = ["lesson","student","skill","cell","right","eol"]
    newset.sort_values(["skill","student"], inplace = True)
    newset.loc[:,'skill'] = newset.loc[:,'skill'].str.replace(r"\s", "")
    r='[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    newset.loc[:,'skill'] = newset.loc[:,'skill'].str.replace(r, "")
    return newset
    newset.to_csv("BKT1_test.txt", sep = "\t", columns = header)'''
####################################################################################################
    
def BKT_Calculator(TEST_Slip):
    TEST_Slip = pd.merge(TEST_Slip, Skill_parameters, how = 'left')
    TEST_Slip.Results = TEST_Slip.Results.astype(int)
    Ln_1 = [0]*len(TEST_Slip)
    Ln_1_Result = [0]*len(TEST_Slip)
    Ln = [0]*len(TEST_Slip)
    L0 = list(TEST_Slip.loc[:,'L0'])
    S = list(TEST_Slip.loc[:,'S'])
    G = list(TEST_Slip.loc[:,'G'])
    T = list(TEST_Slip.loc[:,'T'])
    Results = list(TEST_Slip.loc[:,'Results'])
    skill = list(TEST_Slip.loc[:,'skill'])
    student = list(TEST_Slip.loc[:,'student'])
    Reversed_T = [1-item for item in T]
    Reversed_S = [1-item for item in S]
    Reversed_G = [1-item for item in G]
    TRIO = ["."]*len(TEST_Slip)
    LASTTWO = ['.']*len(TEST_Slip)
    P_TRIO_Ln = [0]*len(TEST_Slip)
    P_TRIO_non_Ln = [0]*len(TEST_Slip)
    P_TRIO = [0]*len(TEST_Slip)
    P_Ln_TRIO = [0]*len(TEST_Slip)
    Guess = [0]*len(TEST_Slip)
    Carelessness = [0]*len(TEST_Slip)
        
    for i in iter(range(0,len(TEST_Slip))):
        if (skill[i] != skill[i-1]) | (student[i] != student[i-1]):
            Ln_1[i] =  L0[i]
        else:
            Ln_1[i] = Ln[i-1]
        if Results[i] == 1:
            Ln_1_Result[i] = ((Ln_1[i]*(1-S[i]))/((Ln_1[i]*(1-S[i]))+((1-Ln_1[i])*G[i])))
        else:
            Ln_1_Result[i] = ((Ln_1[i]*(S[i]))/((Ln_1[i]*(S[i]))+((1-Ln_1[i])*(1-G[i]))))
        Ln[i] = Ln_1_Result[i] + (1-Ln_1_Result[i])*T[i]
       
    for i in iter(range(0,len(TEST_Slip)-2)):
        if student[i] != student[i+2]:
            TRIO[i] = '.'
        else:
            if (Results[i] == 1) & (Results[i+1] == 1) & (Results[i+2] == 1):
                TRIO[i] = 'RRR'
            elif (Results[i] == 1) & (Results[i+1] == 1) & (Results[i+2] == 0):
                TRIO[i] = 'RRW'
            elif (Results[i] == 1) & (Results[i+1] == 0) & (Results[i+2] == 1):
                TRIO[i] = 'RWR'
            elif (Results[i] == 1) & (Results[i+1] == 0) & (Results[i+2] == 0):
                TRIO[i] = 'RWW'
            elif (Results[i] == 0) & (Results[i+1] == 1) & (Results[i+2] == 1):
                TRIO[i] = 'WRR'
            elif (Results[i] == 0) & (Results[i+1] == 1) & (Results[i+2] == 0):
                TRIO[i] = 'WRW'
            elif (Results[i] == 0) & (Results[i+1] == 0) & (Results[i+2] == 1):
                TRIO[i] = 'WWR'
            elif (Results[i] == 0) & (Results[i+1] == 0) & (Results[i+2] == 0):
                TRIO[i] = 'WWW'
                
    for i in iter(range(0,len(TEST_Slip)-2)):
        if student[i] != student[i+2]:
            LASTTWO[i] = '.'
        elif student[i] == student[i+2]:
            if (Results[i+1] == 1) & (Results[i+2] == 1):
                LASTTWO[i] = 'RR'
            elif (Results[i+1] == 1) & (Results[i+2] == 0):
                LASTTWO[i] = 'RW'
            elif (Results[i+1] == 0) & (Results[i+2] == 1):
                LASTTWO[i] = 'WR'
            elif (Results[i+1] == 0) & (Results[i+2] == 0):
                LASTTWO[i] = 'WW'
         
    for i in iter(range(0,len(TEST_Slip))):
        if LASTTWO[i] == '.':
            P_TRIO_Ln[i] = np.nan
        else:
            if LASTTWO[i] == 'RR':
                P_TRIO_Ln[i] = Reversed_S[i]*Reversed_S[i]
            elif LASTTWO[i] == 'RW':
                P_TRIO_Ln[i] = Reversed_S[i]*S[i]
            elif LASTTWO[i] == 'WR':
                P_TRIO_Ln[i] = Reversed_S[i]*S[i]
            elif LASTTWO[i] == 'WW':
                P_TRIO_Ln[i] = S[i]*S[i]
                
    for i in iter(range(0,len(TEST_Slip))):
        if LASTTWO[i] == '.':
            P_TRIO_non_Ln[i] = np.nan
        else:
            if LASTTWO[i] == 'RR':
                P_TRIO_non_Ln[i] = ((Reversed_T[i]**2)*(G[i]**2)) + (Reversed_T[i]*T[i]*\
                G[i]*Reversed_S[i]) + (T[i]*(Reversed_S[i]**2))
            elif LASTTWO[i] == 'RW':
                P_TRIO_non_Ln[i] = ((Reversed_T[i]**2)*G[i]*Reversed_G[i]) + \
                (Reversed_T[i]*T[i]*G[i]*S[i]) + (T[i]*Reversed_S[i]*S[i])
            elif LASTTWO[i] == 'WR':
                P_TRIO_non_Ln[i] = ((Reversed_T[i]**2)*Reversed_G[i]*G[i]) + \
                (Reversed_T[i]*T[i]*Reversed_G[i]*Reversed_S[i]) + \
                (T[i]*S[i]*Reversed_S[i])
            elif LASTTWO[i] == 'WW':
                P_TRIO_non_Ln[i] = ((Reversed_T[i]**2)*(Reversed_G[i]**2)) + \
                (Reversed_T[i]*T[i]*Reversed_G[i]*S[i]) + (T[i]*(S[i]**2))
    
    for i in iter(range(0,len(TEST_Slip))):
        if TRIO[i] == '.':
            P_TRIO[i] = np.nan
        else:
            P_TRIO[i] = (Ln[i]*P_TRIO_Ln[i]) + ((1-P_TRIO_Ln[i])*P_TRIO_non_Ln[i])
    
    for i in iter(range(0,len(TEST_Slip))):
        if TRIO[i] == '.':
            P_Ln_TRIO[i] = '.'
        else:
            P_Ln_TRIO[i] = (P_TRIO_Ln[i]*Ln[i])/P_TRIO[i]
    
    for i in iter(range(0,len(TEST_Slip))):
        if (TRIO[i] == '.') | (Results[i] == 0):
            Guess[i] = np.nan
        else:
            Guess[i] = 1-P_Ln_TRIO[i]
    
    for i in iter(range(0,len(TEST_Slip))):
        if (TRIO[i] == '.') | (Results[i] == 1):
            Carelessness[i] = np.nan
        else:
            Carelessness[i] = P_Ln_TRIO[i]
    
    TEST_Slip['Ln-1'] = Ln_1
    TEST_Slip['Ln-1_Result'] = Ln_1_Result
    TEST_Slip['Ln'] = Ln   
    TEST_Slip['Guess'] = Guess
    TEST_Slip['Carelessness'] = Carelessness
    return TEST_Slip

  ##################################################################################################          
            
# Calculate the correctness of items
def Correctness_of_items(BKT):
    BKT.Results = BKT.Results.astype(int)
    ItemCorrectness = BKT.groupby('item').Results.mean()
    ItemCorrectness = pd.DataFrame(ItemCorrectness)
    ItemCorrectness.rename(columns = {'Results':'Item_correctness'}, inplace = True) 
    ItemCorrectness.reset_index(inplace = True)
    return ItemCorrectness
    
####################################################################################################

def Create_streakcolumn(RMData):
    RollingSet = RMData.loc[:, ['student', 'Datetime', 'skill', 'item', 'TryNumber', 'Result', 'Hints']]
    RollingSet = RollingSet.loc[((RollingSet['Result'] == 'correct') | (RollingSet['Result'] == 'incorrect')),:]
    #RollingSet.Datetime = pd.to_datetime(RMData.Datetime, utc = -6)
    RollingSet.sort_values(["student","Datetime"], inplace = True)
    RollingSet['Results_WithoutHints'] = np.where((RollingSet.Result == 'incorrect')|(RollingSet.Hints == 1.0), 0, 1)
    RollingSet['Streak'] = RollingSet.Results_WithoutHints.rolling(5).sum()
    Concat_RollingSet = RollingSet.loc[:,'Streak']
    Final = pd.concat([RMData, Concat_RollingSet], axis = 1)
    return Final

####################################################################################################
    
def Aggregate_RMData(RMData):
    Aggregate = RMData.groupby('student', as_index = False).\
    agg({'SchoolID':'max', 'TeacherID': 'max', 'GradeLevel': 'max', \
     'Datetime':'min', 'TimeUsage': 'sum','Result_of_speedgame':'mean',\
     'Result_of_theory':'mean','Result_of_notestest':'mean', \
     'Result_of_ProblemA':'mean','Result_of_ProblemB':'mean', \
     'Result_of_ProblemC':'mean', 'If_RMG':'sum',\
     'WallofMastery_ProblemC_time':'sum', 'If_myplace_20s':'sum',\
     'Myplace_20s_time': 'sum', 'Time_at_home':'sum',\
     'Wrongactions_overthan6s':'sum', 'Nohelp_poorknown':'sum',\
     'Nohelp_wellknown':'sum', 'Solve_difficult_items':'sum',\
     'Cost_on_books':'sum', 'Time_after_incorrect':'mean',\
     'LongP_afterhints':'sum', 'ShortP_afterhints':'sum',\
     'LongP_afterhints_correct':'sum', 'ShortP_afterhints_correct':'sum',\
     'OffTask_AfterStreak':'max', 'SwitchMode_AfterStreak':'sum',\
     'Guess':'mean', 'Carelessness':'mean', 'Speedgame_totaltime':'sum',\
     'Speedgame_totalitems':'sum','Basic_practice_time':'sum', 'If_speedgame':'sum',\
     'If_theory':'sum', 'If_notestest':'sum', 'If_ProblemA':'sum',\
     'If_ProblemB':'sum','If_ProblemC':'sum'}).\
    rename(columns = {'Result_of_speedgame':'Correctness_speedgame',\
                       'Result_of_theory':'Correctness_theory',\
                       'Result_of_notestest':'Correctness_notestest',\
                       'Result_of_ProblemA':'Correctness_ProblemA',\
                       'Result_of_ProblemB':'Correctness_ProblemB',\
                       'Result_of_ProblemC':'Correctness_ProblemC',\
                       'If_RMG':'RMG_actions',\
                       'If_myplace_20s':'DecoratingRoom_actions',\
                       'Myplace_20s_time':'DecoratingRoom_time', 'Guess':'AvgGuess',\
                       'Carelessness':'AvgCarelessness',\
                       'Time_after_incorrect':'AvgTime_after_incorrect',\
                       'TimeUsage':'Total_Time'})

    Aggregate['RMG_actions'] = Aggregate.RMG_actions.fillna(0)
    Aggregate['If_speedgame'] = Aggregate.If_speedgame.fillna(0)
    Aggregate['If_theory'] = Aggregate.If_theory.fillna(0)
    Aggregate['If_notestest'] = Aggregate.If_notestest.fillna(0)
    Aggregate['If_ProblemA'] = Aggregate.If_ProblemA.fillna(0)
    Aggregate['If_ProblemB'] = Aggregate.If_ProblemB.fillna(0)
    Aggregate['If_ProblemC'] = Aggregate.If_ProblemC.fillna(0)
    Total_Actions = RMData.student.value_counts()
    Total_Actions = pd.DataFrame(Total_Actions)
    Total_Actions.reset_index(inplace = True)
    Total_Actions = Total_Actions.rename(columns = {'index':'student', 'student':'Total_Actions'})
    Aggregate = pd.merge(Aggregate, Total_Actions, on = 'student', how = 'left')
    Aggregate['Total_LearningActions'] = Aggregate['If_speedgame'] + Aggregate['If_theory']\
    + Aggregate['If_notestest'] + Aggregate['If_ProblemA'] + Aggregate['If_ProblemB']\
    + Aggregate['If_ProblemC']
    Aggregate['PropTime_WallofMastery_ProblemC'] = Aggregate.WallofMastery_ProblemC_time\
    /Aggregate.Total_Time
    Aggregate['MathRace_Speed'] = Aggregate.Speedgame_totaltime/Aggregate.Speedgame_totalitems
    Aggregate['PropActions_RMG'] = Aggregate.RMG_actions/Aggregate.Total_Actions
    Aggregate['PropTime_BasicPratice'] = Aggregate.Basic_practice_time/Aggregate.Total_Time
    Aggregate['PropActions_DecoratingRoom'] = Aggregate.DecoratingRoom_actions\
    /Aggregate.Total_Actions
    Aggregate['PropTime_Home'] = Aggregate.Time_at_home/Aggregate.Total_Time
    Aggregate['PropActions_Nohelp_Wellknown'] = Aggregate.Nohelp_wellknown\
    /Aggregate.Total_Actions
    Aggregate['PropActions_Nohelp_poorknown'] = Aggregate.Nohelp_poorknown\
    /Aggregate.Total_Actions
    Aggregate['PropActions_LongP_Afterhints'] = Aggregate.LongP_afterhints\
    /Aggregate.Total_Actions
    Aggregate['PropActions_ShortP_Afterhints'] = Aggregate.ShortP_afterhints\
    /Aggregate.Total_Actions
    Aggregate['PropActions_LongP_Afterhints_correct'] = Aggregate.LongP_afterhints_correct\
    /Aggregate.Total_Actions
    Aggregate['PropActions_ShortP_Afterhints_correct'] = Aggregate.ShortP_afterhints_correct\
    /Aggregate.Total_Actions
    Aggregate['PropActions_Solve_DifficultItem'] = Aggregate.Solve_difficult_items\
    /Aggregate.Total_Actions
    Aggregate['PropActions_WrongActions>6s'] = Aggregate.Wrongactions_overthan6s\
    /Aggregate.Total_Actions
    Aggregate.drop(['If_speedgame', 'If_theory', 'If_notestest', 'If_ProblemA','If_ProblemB',\
                    'If_ProblemC','Speedgame_totaltime','Speedgame_totalitems'],\
    axis = 1, inplace = True)
    return Aggregate

####################################################################################################

# Concat all the aggregate dataset
RM_AggregateData = pd.concat([Aggregate_01,Aggregate_02_A,Aggregate_02_B,Aggregate_03_A,Aggregate_03_B,Aggregate_04_A,Aggregate_04_B,Aggregate_05_A,Aggregate_05_B,Aggregate_06_A,Aggregate_06_B,Aggregate_07_A,Aggregate_07_B,Aggregate_08_A,Aggregate_08_B,Aggregate_09_A,Aggregate_09_B, Aggregate_10_A,Aggregate_10_B,Aggregate_11, Aggregate_12], axis = 0)

RM_AggregateData['Datetime'] = pd.to_datetime(RM_AggregateData.Datetime)
RM_AggregateData = RM_AggregateData.sort_values(['student', 'Datetime'])

# Examine the number of students
len(RM_AggregateData.student.unique())

# Whether the student write more than 50 words
Fifty_words = pd.read_csv("WriteGenieMail_50Words.csv")


RM_AggregateData = pd.merge(RM_AggregateData, Fifty_words, on = "student", how = 'left', sort = False)

RM_AggregateData['Mail>50words'] = RM_AggregateData['Mail>50words'].fillna(0)

# Extract Date in the Final Aggregate Data
RM_AggregateData['Datetime'] = RM_AggregateData['Datetime'].dt.date

####################################################################################################


# Concat Survey and log file
Survey = pd.read_csv("Survey_spread.csv")
Survey = Survey.drop('Unnamed: 0',axis = 1)
Survey['value'] = Survey['1'] + Survey['2'] + Survey['3'] + Survey['4'] + Survey['5']
Survey['self_efficacy'] = Survey['6'] + Survey['7'] + Survey['8'] + Survey['9'] + Survey['10']
Survey['interest'] = Survey['15'] + Survey['16'] + Survey['17']
Survey_join = Survey.loc[:,['studentID','value', 'self_efficacy','interest']]
Survey_join = Survey_join.rename(columns = {'studentID':'student'}, inplace = True)
RM_AggregateData = pd.merge(RM_AggregateData, Survey_join, how = 'left', on = 'student', sort = False)

####################################################################################################
