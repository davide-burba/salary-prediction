
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
      el("result").innerHTML = `Risultato:<br><br> ${response["prediction"]} \u20AC/anno <br> ${response["monthly_prediction"]} \u20AC/mese`;
    }
    el("predict-button").innerHTML = "Valuta";
  };

  var contract_time = el("contract_time").value;
  var category = el("category").value;
  var region = el("region").value;

  var fileData = new FormData();
  fileData.append("contract_time", contract_time);
  fileData.append("category", category);
  fileData.append("region", region);

  xhr.send(fileData);
}
