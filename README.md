# thesis-data-generator

## Opis danych wejściowych
Dane wejściowe są przechowywane w pliku JSON o nazwie "input". Zawierają listę obiektów, opisujących poszczególne wartości parametrów wejściowych generatora danych. Innymi słowy, każdy obiekt ma następującą strukturę:

```json
{
  "n_eff": ...,
  "grating_period": ...,
  "mode_coupling_coef": ...
}
```

Gdzie:
+ *n_eff* - efektywny współczynnik,
+ *grating_period* - okres siatki,
+ *mode_coupling_coef* - współczynnik sprzężenia,

Dodatkowo, zakłada się, że długość siatki jest stała i wynosi `90μm`.

## Wzory
Wszelkie wzory, założenia wykorzystane w generacji danych zostały wzięte z https://core.ac.uk/display/8986606?utm_source=pdf&utm_medium=banner&utm_campaign=pdf-decoration-v1 
