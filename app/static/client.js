
var el = x => document.getElementById(x);


function predict() {
  
  el("predict-button").innerHTML = "Valutando...";
  var xhr = new XMLHttpRequest();
  var loc = window.location;

  xhr.open("POST", `${loc.protocol}//${loc.hostname}:${loc.port}/predict`,true);
  xhr.onerror = function() {
    alert(xhr.responseText);
  };
  xhr.onload = function(e) {
    if (this.readyState === 4) {
      var response = JSON.parse(e.target.responseText);
      //  ----- PROBNN ----- 
      var img = document.getElementById("ImgDistribution");
      img.src = "images/distribution.png"
      el("result").innerHTML = `<br> Salario netto annuale: <p style="color:red;font-size:80px">${response["prediction"]}</p> Range [75%]: <p style="color:red;">${response["lower_bound"]} - ${response["upper_bound"]}</p> Distribuzione:`;
      // ----- LightGBM ----- 
      // el("result").innerHTML = `Salario netto annuale [stima]: ${response["prediction"]}`;
    }
    el("predict-button").innerHTML = "Valuta";
  };

  var contratto = el("contratto").value;
  var ore_settimana = el("ore_settimana").value;
  var dimensioni_azienda = el("dimensioni_azienda").value;
  var settore = el("settore").value;
  var qualifica = el("qualifica").value;
  var titolo_studio = el("titolo_studio").value;
  var tipo_laurea = el("tipo_laurea").value;
  var tipo_diploma = el("tipo_diploma").value;
  var regione = el("regione").value;
  var ampiezza_comune = el("ampiezza_comune").value;
  var anni_da_primo_lavoro = el("anni_da_primo_lavoro").value;
  var anni_da_lavoro_corrente = el("anni_da_lavoro_corrente").value;
  var anni_da_edu = el("anni_da_edu").value;
  var anni_contributi = el("anni_contributi").value;
  var n_esp_lavorative = el("n_esp_lavorative").value;

  var fileData = new FormData();
  fileData.append("contratto", contratto);
  fileData.append("ore_settimana", ore_settimana);
  fileData.append("dimensioni_azienda", dimensioni_azienda);
  fileData.append("settore", settore);
  fileData.append("qualifica", qualifica);
  fileData.append("titolo_studio", titolo_studio);
  fileData.append("tipo_laurea", tipo_laurea);
  fileData.append("tipo_diploma", tipo_diploma);
  fileData.append("regione", regione);
  fileData.append("ampiezza_comune", ampiezza_comune);
  fileData.append("anni_da_primo_lavoro", anni_da_primo_lavoro);
  fileData.append("anni_da_lavoro_corrente", anni_da_lavoro_corrente);
  fileData.append("anni_da_edu", anni_da_edu);
  fileData.append("anni_contributi", anni_contributi);
  fileData.append("n_esp_lavorative", n_esp_lavorative);

  xhr.send(fileData);
}
