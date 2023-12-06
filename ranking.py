'''
Calculates Recall Rate @ k
'''
import datetime
import itertools
import logging
import math
import multiprocessing
import os
import pickle
import random
from datetime import timedelta
from multiprocessing import Queue
from queue import Empty
import numpy as np
import ujson
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

# from data.preprocessing import concatenateSummaryAndDescription
from data_util import readDateFromBug, createChunks
from time import time

class RankingData(object):

    def __init__(self, preprocessingList):
        self.anchor_input = None
        self.bug_ids = None
        self.preprocessingList = preprocessingList

    def reset(self, anchor_input, bug_ids):
        self.anchor_input = anchor_input
        self.bug_ids = bug_ids

    def __len__(self):
        return len(self.bug_ids)

    def __getitem__(self, idx):
        return [self.anchor_input, self.preprocessingList.extract(self.bug_ids[idx]), 0.0]

# Implement the three methods to calculate the recall rate

class SunRanking(object):

    def __init__(self, bugReportDatabase, dataset, window):
        self.bugReportDatabase = bugReportDatabase
        self.masterIdByBugId = self.bugReportDatabase.getMasterIdByBugId()
        self.duplicateBugs = dataset.duplicateIds
        self.candidates = []
        self.window = int(window) if window is not None else 0
        self.latestDateByMasterSetId = {}
        self.logger = logging.getLogger()

        # Get oldest and newest duplicate bug report in dataset
        oldestDuplicateBug = (
            self.duplicateBugs[0], readDateFromBug(self.bugReportDatabase.getBug(self.duplicateBugs[0])))

        for dupId in self.duplicateBugs:
            dup = self.bugReportDatabase.getBug(dupId)
            creationDate = readDateFromBug(dup)

            if oldestDuplicateBug[1] < creationDate:
                oldestDuplicateBug = (dupId, creationDate)

        # Keep only master that are able to be candidate
        for bug in self.bugReportDatabase.bugList:
            bugCreationDate = readDateFromBug(bug)
            bugId = bug['bug_id']

            # Remove bugs that their creation time is bigger than oldest duplicate bug
            if bugCreationDate > oldestDuplicateBug[1] or (
                    bugCreationDate == oldestDuplicateBug[1] and bug['bug_id'] > oldestDuplicateBug[0]):
                continue

            self.candidates.append((bugId, bugCreationDate.timestamp()))

        # Keep the timestamp of all reports in each master set
        for masterId, masterSet in self.bugReportDatabase.getMasterSetById(
                map(lambda c: c[0], self.candidates)).items():
            ts_list = []

            for bugId in masterSet:
                bugCreationDate = readDateFromBug(self.bugReportDatabase.getBug(bugId))

                ts_list.append((int(bugId), bugCreationDate.timestamp()))

            self.latestDateByMasterSetId[masterId] = ts_list

        # Set all bugs that are going to be used by our models.
        self.allBugs = [bugId for bugId, bugCreationDate in self.candidates]
        self.allBugs.extend(self.duplicateBugs)

    def getDuplicateBugs(self):
        return self.duplicateBugs

    def getAllBugs(self):
        return self.allBugs

    def getCandidateList(self, anchorId):
        candidates = []
        anchor = self.bugReportDatabase.getBug(anchorId)
        anchorCreationDate = readDateFromBug(anchor)
        anchorMasterId = self.masterIdByBugId[anchorId]
        nDupBugs = 0
        anchorTimestamp = anchorCreationDate.timestamp()
        anchorDayTimestamp = int(anchorTimestamp / (24 * 60 * 60))

        nSkipped = 0
        window_record = [] if self.logger.isEnabledFor(logging.DEBUG) else None
        anchorIdInt = int(anchorId)

        for bugId, bugCreationDate in self.candidates:
            bugIdInt = int(bugId)

            # Ignore reports that were created after the anchor report
            if bugCreationDate > anchorTimestamp or (
                    bugCreationDate == anchorTimestamp and bugIdInt > anchorIdInt):
                continue

            # Check if the same report
            if bugId == anchorId:
                continue

            if bugIdInt > anchorIdInt:
                self.logger.warning(
                    "Candidate - consider a report which its id {} is bigger than duplicate {}".format(bugId, anchorId))

            masterId = self.masterIdByBugId[bugId]

            # Group all the duplicate and master in one unique set. Creation date of newest report is used to filter the bugs
            tsMasterSet = self.latestDateByMasterSetId.get(masterId)
            # print("tsMasterSet",tsMasterSet)

            if tsMasterSet:
                max = -1
                newest_report = None

                for candNewestId, ts in self.latestDateByMasterSetId[masterId]:
                    # Ignore reports that were created after the anchor or the ones that have the same ts and bigger id
                    if ts > anchorTimestamp or (ts == anchorTimestamp and candNewestId >= anchorIdInt):
                        continue

                    if candNewestId >= anchorIdInt:
                        self.logger.warning(
                            "Window filtering - consider a report which its id {} is bigger than duplicate {}".format(
                                candNewestId,
                                anchorIdInt))

                    # Get newest ones
                    if max < ts:
                        max = ts
                        # print("max ", max)
                        newest_report = candNewestId
                # Transform to day timestamp
                bug_timestamp = int(max / (24 * 60 * 60))
            else:
                # Transform to day timestamp
                bug_timestamp = int(bugCreationDate / (24 * 60 * 60))
                newest_report = bugId

            # Is it in the window?
            if 0 < self.window < (anchorDayTimestamp - bug_timestamp):
                nSkipped += 1
                continue

            # Count number of duplicate bug reports
            if anchorMasterId == masterId:
                nDupBugs += 1

            # It is a candidate
            candidates.append(bugId)

            if window_record is not None:
                window_record.append((bugId, newest_report, bug_timestamp))

        self.logger.debug(
            "Query {} ({}) - window {} - number of reports skipped: {}".format(anchorId, anchorDayTimestamp,
                                                                               self.window, nSkipped))

        if window_record is not None:
            self.logger.debug("{}".format(window_record))

        if nDupBugs == 0:
            return []

        return candidates


class DeshmukhRanking(object):

    def __init__(self, bugReportDatabase, dataset):
        self.bugReportDatabase = bugReportDatabase
        self.masterIdByBugId = self.bugReportDatabase.getMasterIdByBugId()
        self.duplicateBugs = dataset.duplicateIds
        self.allBugs = dataset.bugIds

    def getAllBugs(self):
        return self.allBugs

    def getDuplicateBugs(self):
        return self.duplicateBugs

    def getCandidateList(self, anchorId):
        return [bugId for bugId in self.allBugs if anchorId != bugId]


class PreselectListRanking(object):

    def __init__(self, filePath, nBugToSample=0):
        self.allBugIds, self.listByBugId = pickle.load(open(filePath, 'rb'))
        self.duplicateBugs = list(self.listByBugId.keys())
        self.nBugToSample = len(self.duplicateBugs) if nBugToSample is None or nBugToSample <= 0 else nBugToSample

    def getDuplicateBugs(self):
        if self.nBugToSample == len(self.duplicateBugs):
            return self.duplicateBugs

        return random.sample(self.duplicateBugs, self.nBugToSample)

    def getCandidateList(self, anchorId):
        return self.listByBugId[anchorId]

    def getAllBugs(self):
        return self.allBugIds


class RecallRate(object):

    def __init__(self, bugReportDatabase, k=None, groupByMaster=True):
        self.masterSetById = bugReportDatabase.getMasterSetById()
        self.masterIdByBugId = bugReportDatabase.getMasterIdByBugId()

        if k is None:
            k = list(range(1, 21))

        self.k = sorted(k)

        self.hitsPerK = dict((k, 0) for k in self.k)
        self.nDuplicate = 0
        self.logger = logging.getLogger()

        self.groupByMaster = groupByMaster

    def reset(self):
        self.hitsPerK = dict((k, 0) for k in self.k)
        self.nDuplicate = 0

    def update(self, anchorId, recommendationList):
        mastersetId = self.masterIdByBugId[anchorId]
        masterSet = self.masterSetById[mastersetId]
        # biggestKValue = self.k[-1]

        # pos = biggestKValue + 1
        pos = math.inf
        correct_cand = None

        if len(recommendationList) == 0:
            self.logger.warning("Recommendation list of {} is empty. Consider it as miss.".format(anchorId))

        else:
            seenMasters = set()

            for bugId, p in recommendationList:
                mastersetId = self.masterIdByBugId[bugId]

                if self.groupByMaster:
                    if mastersetId in seenMasters:
                        continue

                    seenMasters.add(mastersetId)
                else:
                    seenMasters.add(bugId)

                # if len(seenMasters) == pos:
                #     break

                if bugId in masterSet:
                    pos = len(seenMasters)
                    correct_cand = bugId
                    break

        # If one of k duplicate bugs is in the list of duplicates, so we count as hit. We calculate the hit for each different k
        for idx, k in enumerate(self.k):
            if k < pos:
                continue

            self.hitsPerK[k] += 1

        self.nDuplicate += 1

        return pos, correct_cand

    def compute(self):
        recallRate = {}
        for k, hit in self.hitsPerK.items():
            rate = float(hit) / self.nDuplicate
            recallRate[k] = rate

        return recallRate


class RankingResultFile(object):

    def __init__(self, filePath, bugReportDatabase):
        filename, ext = os.path.splitext(filePath)
        self.filepath = "{}_{}{}".format(filename, int(random.randint(0, 10000000)), ext)
        self.file = open(self.filepath, 'w')
        self.logger = logging.getLogger(__name__)

        self.logger.info({"result_path":self.filepath})

    def update(self, anchorId, recommendationList, pos, correct_cand):
        self.file.write(anchorId)

        for cand_id, score in recommendationList:
            self.file.write(" ")
            self.file.write(cand_id)
            self.file.write("|")
            self.file.write(str(round(score, 3)))

        self.file.write("\n")
        self.file.flush()
