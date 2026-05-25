import numpy as np
import ot #pip install POT
#idh 1 = mutant
class Participant: #PatientID SampleID DiagnosisAge ATRXstatus BCRStatus BRAF-KIAA1549fusion BRAFV600Estatus CancerType CancerTypeDetailed Chr19/20co-gain Chr7gain/Chr10loss ESTIMATEcombinedscore ESTIMATEimmunescore ESTIMATEstromalscore NeoplasmHistologicGrade IDH/codelsubtype IDH-specificDNAMethylationCluster IDH-specificRNAExpressionCluster IDH KarnofskyPerformanceScore MGMT MutationCount MONTHS Status Pan-GliomaDNAMethylationCluster Pan-GliomaRNAExpressionCluster Percentaneuploidy AbsolutePurity RandomForestSturmCluster Sex SupervisedDNAMethylationCluster Telomerelengthestimateinbloodnormal(Kb) Telomerelengthestimateintumor(Kb) TelomereMaintenance TERTexpression(log2) TERTexpressionstatus TERTpromoterstatus TMB(nonsynonymous) TranscriptomeSubtype EGFR 
                   #required to be uppercase
  def __init__(self):
    self.values = dict()
  def initalize(self, headers, values):
    hds = headers.split(" ")
    vals = values.split(" ")
    self.values = dict()

    for i in range(len(hds)):
      self.values[hds[i]] = vals[i]

  def deepcopy(self):
    c = dict()
    p = Participant()
    for i in self.values:
      c[i] = self.values[i]
    p.values = c
    return p

  def test(self, a, b, c, d, e):
        self.values = dict()
        self.values["IDH"] = a
        self.values["EGFR"] = c
        self.values["MGMT"] = b
        self.values["GRADE"] = e
        self.values["MONTHS"] = d

class Consistancy:
    def rDR(self, idh):
        return 1 - idh
    def rDC(self, idh, mgmt):
        if(idh == 0 or mgmt == 0):
            return 1
        else:
            return 0
    def rRC(self, egfr):
        return egfr
    def severity(self, grade, months):
        if(grade == 1):
            return 1
        if(months < 24):
            return 1
        return 0
    def eDR(self, idh, egfr):
        return abs(self.rDR(idh) - egfr)
    def eDC(self, idh, mgmt, grade, months):
        return abs(self.rDC(idh, mgmt) - self.severity(grade, months))
    def eRC(self, egfr, grade, months):
        return abs(self.rRC(egfr) - self.severity(grade, months))
    
    def consistancyVector(self, p):
        vals = p.vals
        idh = vals["IDH"] 
        egfr = vals["EGFR"]
        mgmt = vals["MGMT"]
        grade = vals["GRADE"]
        surv = vals["MONTHS"]
        eDR = self.eDR(idh, egfr)
        eDC = self.eDC(idh, mgmt, grade, surv)
        eRC = self.eRC(egfr, grade, surv)

        return str(eDR) + " " + str(eDC) + " " + str(eRC)
    
    def consistancy(self, p):
      str = self.consistancyVector(p)
      sum = 0
      for i in str:
         if(i == '1'):
            sum += 1
      return sum/3

class EncodedCases:
  #Use raw molecular call, ignore WHO text label if conflicting
  #Exclude sample if missing (required for subtype classification)
  def IDHMutationStatus(self, par):
    if(par.values.get("IDH") is None):
      return None
    p = par.deepcopy()
    
    if(par.values["IDH"] == "mutant"):
      p.values["IDH"] = 1
    else:
      p.values["IDH"] = 0
    return p

  #Binarize from beta value: threshold ≥ 0.30 = methylated
  #If MGMT data is missing, fill it in with whatever value appears most in the cohort, 
  #and mark those samples so we know they were filled in and not real measurements
  def MGMTMethylation(self, par):
    p = par.deepcopy()
    if(par.values.get("MGMT") is None):
      print("remind michael that he still has to do this part, and is just waiting to see what the data looks like")
    elif(par.values["MGMT"] >= 0.3):
      p.values["MGMT"] = 1
    else:
      p.values["MGMT"] = 0
    return p

  #Log2(FPKM+1) normalize; amplified = CNV≥2 OR FPKM ≥ 90th pct
  #Exclude sample from transcriptomic node if RNA-seq missing
  def EGFRExpression(self, par):
    print("")

  #Convert days to months if needed (÷30.44), floor at 0
  #Exclude if OS completely absent; note censored vs. deceased separately
  def overallSurvival(self, par):
    p = par.deepcopy()
    if(p.values.get("MONTHS")is None):
        return None
    if(p.values["MONTHS"] < 24):
       p.values["MONTHS"] = 0
    else:
       p.values["MONTHS"] = 1
    return p
    

  #Standardize: 1=Deceased, 0=Living; resolve string variants
  #Exclude if missing (needed for survival interpretation)
  def vitalStatus(self, par):
    p = par.deepcopy()
    if(p.values.get("STATUS") is None):
       return None
    print(p.values["STATUS"] == 0)
    if(p.values["MONTHS"] == 0 and p.values["STATUS"] == "0"):
       return None
    return p
  
  #Map numeric: 2→2, 3→3, 4→4; cross check IDH for label consistency
  #Use IDH status to infer if WHO label absent or ambiguous
  def WHOGrade(self, par):
    print("")

class OTOutputs: #change topics in data, its hard coded...
  def __init__(self, dat):
    self.data = dat
    self.avgs = dat.avgs
    
  def omicsDistance(self, i, j):
    pars = [i, j]
    vals = dict()
    
    for t in self.data.topics:
      if(not i.values[t] == "NA" and j.values[t] == "NA"):
        if(i.values[t] == "NA" or j.values[t] == "NA"):
          if(i.values[t] == "NA"):
            vals[t] = j.values[t]
          else:
            vals[t] = i.values[t]
        else:
          vals[t] = abs(i.values[t] - j.values[t])

    cost = 0
    for c in vals:
      cost += float(vals[c]) * float(vals[c])
      
    return cost
  
  def IDHPenelty(self, i, j):
    if(i.values["IDH"] == j.values["IDH"]):
      return 0
    return 1
  
  def MGMTPenelty(self, i, j):
    if(i.values["MGMT"] == j.values["MGMT"]):
      return 0
    return 1
    
  def gradePenelty(self, i, j):
    str = "NeoplasmHistologicGrade".upper()
    return abs(int((i.values[str])[1:2]) - int((j.values[str])[1:2]))
  
  def agePenelty(self, i, j):
    return abs(i.values["DIAGNOSISAGE"] - j.values["DIAGNOSISAGE"])

  def purityPenelty(self, i, j):
    str = "AbsolutePurity".upper()
    try:
      n = float(i.values[str]) - float(j.values[str])
      return abs(n)
    except ValueError:
      return -1
  
  def cost(self, i, j):
    omics = self.omicsDistance(i, j)
    idh = self.IDHPenelty(i, j) * 0.5
    mgmt = self.MGMTPenelty(i, j) * 0.5
    grade = self.gradePenelty(i, j) * 0.25
    purity = self.purityPenelty(i, j) * 0.4
    
    return omics + idh + mgmt + grade + purity
  
  def costMatrix(self):
    length = len(self.data.vals)
    arr = np.zeros((length, length))
    for r in range(0, length):
      for c in range(0, length):
        v = self.data.vals
        arr[r,c] = self.cost(v[r], v[c])
    return arr
  
  def result(self):
    c = self.costMatrix()
    return ot.solve_sample(c, c)
  
  def resultMatrx(self):
    return self.result().plan
    
class input:
    def __init__(self):
      self.vals = []
    
    def infile(self, name):
      file = open(name)
      self.head = file.readline().upper().split()
      
      for line in file:
          p = Participant()
          vls = line.split()
          for i in range(len(vls)):
            p.values[self.head[i]] = vls[i]
          self.vals.append(p)
      return self.vals
         
class data: #change topics for omics
  def __init__(self, filename):
    self.avgs = dict()
    self.reader = input()
    self.vals = self.reader.infile("data.txt")
    self.topics = ["DIAGNOSISAGE", "ABSOLUTEPURITY", "MONTHS"] 
    
    for t in self.topics:
      count = 0;
      sum = 0;
      for i in self.vals:
        try:
          sum += float(i.values[t])
          count += 1
        except ValueError:
          count = count #just a place holder, does nothing
      self.avgs[t] = sum/count
          
          
          
def rainIkUrDementiaSoThisWillPrintTheOTMatrixRawToAFileTheThingIfYouTypeThisOutLMAO(d): #d = data prints to rainurdementia.txt
  with open("rainurdementia.txt", "w") as f:
    re = OTOutputs(d).result().plan
    for r in range(0, len(d.vals)):
      str2 = ""
      for c in range(0, len(d.vals)):
        str2 += str(re[r][c]) + " "
      f.write(str2 + "\n")
       
  



d = data("data.txt")
o = OTOutputs(d)
re = o.result().plan

rainIkUrDementiaSoThisWillPrintTheOTMatrixRawToAFileTheThingIfYouTypeThisOutLMAO(d)
