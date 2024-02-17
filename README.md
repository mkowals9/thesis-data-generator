# thesis-data-generator

## Opis danych wejściowych
Dane wejściowe są przechowywane w pliku JSON o nazwie "input". Zawierają listę obiektów, opisujących poszczególne wartości parametrów wejściowych generatora danych. Innymi słowy, każdy obiekt ma następującą strukturę:

```json
{
  "n_eff": ...,
  "grating_period": ...,
  "delta_n_eff": ...
}
```

Gdzie:
+ *n_eff* - efektywny współczynnik,
+ *grating_period* - okres siatki,
+ *delta_n_eff* - współczynnik "dc" zmiany efektywnego współczynnika uśrednionego na cały okres siatki (“dc” index change spatially averaged over a grating period),
  

Dodatkowo, zakłada się, że długość siatki jest stała i wynosi `4mm`.

## Wzory
Wszelkie wzory, założenia wykorzystane w generacji danych zostały wzięte z artykułu "Fiber Grating Spectra" (Turan Erdogan, IEEE)
