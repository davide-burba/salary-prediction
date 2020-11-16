import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse
import asyncio
from pydantic import BaseModel
import os
import sys
import io

sys.path.insert(0,"src/")


catfeat_map = dict(
    partime = {"a tempo pieno": 1, "part-time": 2},
    contratto = {"a tempo indeterminato" : 1.0, "a tempo determinato" : 2.0, "di lavoro interinale" : 3.0, "non specificato" : None},
    dimensioni_azienda = {"fino a 4 addetti": 1, "tra 5 e 15 addetti": 2, "tra 16 e 19 addetti": 3, "tra 20 e 49 addetti": 4, "tra 50 e 99 addetti": 5, "tra 100 e 499 addetti": 6, "500 addetti ed oltre": 7, "Pubblica Amministrazione": 8, "non specificato":  None},
    titolo_studio = {"nessuno" : 1,"licenza elementare" : 2,"licenza media inferiore" : 3,"diploma professionale (3 anni)" : 4,"diploma media superiore" : 5,"diploma universitario/laurea triennale" : 6,"laurea/laurea magistrale" : 7,"specializzazione post-laurea" : 8},
    tipo_laurea = {"matematica, fisica, chimica, biologia, scienze, farmacia" : 1,"scienze agrarie e veterinaria" : 2,"medicina e odontoiatria" : 3,"ingegneria" : 4,"architettura e urbanistica" : 5,"economia e statistica" : 6,"scienze politiche, sociologia" : 7,"giurisprudenza" : 8,"lettere, filosofia, lingue, pedagogia, psicologia" : 9,"altro" : 10,"non laureato" : None,},
    tipo_diploma = {"istituto professionale" : 1,"istituto tecnico" : 2,"liceo (classico, scientifico e linguistico)" : 3,"liceo artistico e istituti d’arte" : 4,"magistrali" : 5,"altro" : 6,"non specificato" : None,},
    qualifica = {"operaio o posizione similare (inclusi salariati e apprendisti, lavoranti a domicilio, commessi)" : 1,"impiegato" : 2,"insegnante di qualunque tipo di scuola (inclusi incaricati, contrattisti e simili)" : 3,"impiegato direttivo/quadro" : 4,"dirigente, alto funzionario, preside, direttore didattico, docente universitario, magistrato" : 5,"altro" : 21,},
    settore = {"Agricoltura, silvicoltura e pesca" : 1, "Attività estrattive" : 2, "Attività manifatturiere" : 3, "Fornitura di energia elettrica, gas, vapore e aria condizionata" : 4, "Fornitura di acqua; reti fognarie, attività di trattamento dei rifiuti e risanamento" : 5, "Costruzioni" : 6, "Commercio all’ingrosso e al dettaglio; riparazioni di autoveicoli e motocicli" : 7, "Trasporto e magazzinaggio" : 8, "Servizi di alloggio e di ristorazione" : 9, "Servizi di informazione e comunicazione" : 10, "Attività finanziarie e assicurative" : 11, "Attività immobiliari" : 12, "Attività professionali, scientifiche e tecniche" : 13, "Attività amministrative e di servizi di supporto" : 14, "Amministrazione pubblica e difesa; assicurazione sociale obbligatoria" : 15, "Istruzione" : 16, "Sanità e assistenza sociale" : 17, "Attività artistiche, di intrattenimento e divertimento" : 18, "Altre attività di servizi" : 19, "Attività di famiglie e convivenze come datori di lavoro per personale domestico; produzione di beni e servizi indifferenziati per uso proprio da parte di famiglie e convivenze" : 20, "Attività di organizzazioni e organismi extraterritoriali" : 21, "non specificato" : None},
    regione = {"Piemonte" : 1,"Valle d'Aosta" : 2,"Lombardia" : 3,"Trentino" : 4,"Veneto" : 5,"Friuli" : 6,"Liguria" : 7,"Emilia Romagna" : 8,"Toscana" : 9,"Umbria" : 10,"Marche" : 11,"Lazio" : 12,"Abruzzo" : 13,"Molise" : 14,"Campania" : 15,"Puglia" : 16,"Basilicata" : 17,"Calabria" : 18,"Sicilia" : 19,"Sardegna" : 20,},
    ampiezza_comune = {"fino a 5.000 abitanti" : 1, "5.000-20.000 abitanti" : 2, "20.000-50.000 abitanti" : 3, "50.000-200.000 abitanti" : 4, "oltre 200.000 abitanti" : 5, }
)

path = os.path.dirname(os.path.abspath(__file__)) + "/"
app = FastAPI()
app.mount("/static", StaticFiles(directory=path+"static"), name="static")
app.mount("/images", StaticFiles(directory=path+"images"), name="images")


model = pickle.load(open(path + "../models/20201116/model.p","rb"))
features = pickle.load(open(path + "../models/20201116/features.p","rb"))
preprocessor = pickle.load(open(path + "../models/20201116/cat_encoder.p","rb"))

@app.route('/')
async def homepage(request):
    html_file = Path(path + 'view/index.html')
    return HTMLResponse(html_file.open().read())


@app.route('/predict', methods=['POST'])
async def predict(request):
    # get data
    tmp = await request.form()
    x = {f : tmp[f] for f in features}
    # format
    for f in catfeat_map: 
        x[f] = catfeat_map[f][x[f]]
    x = pd.DataFrame(x, index=[0])     
    x = preprocessor.transform(x)
    # predict
    mu,sigma = model.predict_parameters(x)
    mu = mu.item()
    sigma = sigma.item()
    fig = show_normal(mu,sigma)
    pred = mu * model.y_unit
    fig_stream = get_figure(mu,sigma)

    return JSONResponse({
        "prediction" : np.round(pred),
        "monthly_prediction" : np.round(pred/12),
        "mu" : mu,
        "sigma": sigma,
        #"fig_stream":fig_stream, # Object of type StreamingResponse is not JSON serializable
    })


#@app.route("/vector_image", methods=['POST'])
def get_figure(mu,sigma):

    print(mu,sigma)
    fig = show_normal(mu,sigma)
    buf = io.BytesIO()
    fig.savefig(buf, format = 'png')
    return StreamingResponse(buf, media_type="image/png")


def show_normal(mu,sigma,perc = .75,y_unit = 1000):
    # set x and y
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    y = stats.norm.pdf(x, mu, sigma)

    perc = .75
    start,end = stats.norm.interval(perc,mu,sigma)
    x_int = np.linspace(start,end, 100)
    y_int = stats.norm.pdf(x_int, mu, sigma)
    ticks = [start,mu,end]

    # plot
    with plt.style.context('dark_background'):
        fig = plt.figure(figsize = (12,6))
        plt.fill_between(x,y, color = "C0", alpha = .5)
        plt.fill_between(x_int,y_int, label = "75%", color = "C0", alpha=.8)
        density_mu = stats.norm.pdf(mu, mu, sigma)
        plt.text(mu,density_mu * .3,
                 "  {}%".format(int(100*perc)),
                 size = 40,
                 horizontalalignment='center',
                 color = "gray")
        plt.text(mu,density_mu*1.08,
                 "{:,}€".format(int((mu * y_unit))).replace(",","."),
                 size = 70,
                 horizontalalignment='center',
                 color = "C3")
        plt.xticks(ticks,["{:,}€".format(int((v * y_unit))).replace(",",".") for v in ticks],size=25)
        plt.yticks([])
        plt.ylim(0,density_mu*1.4)
        plt.xlim(mu - 3*sigma, mu + 3*sigma)
        plt.title("Salario netto annuale [stima]",size = 30)
    return fig