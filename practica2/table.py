# Generador de tablas de latex

#k-means 1
#cal=[3042.32565654035, 4703.276490723743, 4136.87342581802, 4038.971966681051, 3853.2163277602067, 4151.423190642611, 4226.279512646909, 4805.497922282589]
#sil=[0.5541358699263358, 0.6436774751523562, 0.657690369714004, 0.6891979028078193, 0.7027316444530789, 0.7368975382285727, 0.7867435070579235, 0.8417025367184847]
# ward1
sil=[0.5539252864441229, 0.6001580718033604, 0.6236197963073052, 0.6674109841636733, 0.7150230980315533, 0.7540894010176521, 0.7905300742729079, 0.8254082855580414]
cal=[3038.939895342106, 3263.5006108555554, 3211.2214318407155, 3504.659880500535, 3925.906376649705, 3951.377474318098, 4090.581508592909, 4256.869613116355]
K=list(range(2,10))


# K-means y Wardl

cal=[round(c,2) for c in cal]
sil=[round(s,4) for s in sil]

for i in range(len(K)):
    print(str(K[i])+' & '+str(cal[i])+' & '+str(sil[i])+' \\\\ \\hline')
