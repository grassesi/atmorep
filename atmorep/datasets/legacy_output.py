class LogGlobalForecast(Output):
    def __init__(self, config, model, fields_prediction_idx, epoch):
        self.config = config
        self.model = model
        self.fields_prediction_idx = fields_prediction_idx

        self.levels = np.array(self.config.fields[0][2])
        self.num_tokens = NumTokens(*self.config.fields[0][3])
        self.token_sizes = TokenSize(*self.config.fields[0][4])

        field_names = {field_info[0] for field_info in config.fields}

        writers = {}
        for out_type in OutputType:
            # initializes store
            store = get_store(self.config.wandb_id, epoch, out_type, field_names)
            writers[out_type] = WriterForecast(
                out_type,
                store,
                subset_start_batch_idx=self.model.subset_start_batch_idx,
            )

        super().__init__(writers)

    def log(self, batch_idx, log_sources, log_preds):
        """Logging for BERT_strategy=forecast."""
        cf = self.config
        detok = utils.detokenize

        # save source: remains identical so just save ones
        (sources, token_infos, targets, _, _) = log_sources

        forecast_num_tokens = 1
        if hasattr(cf, "forecast_num_tokens"):
            forecast_num_tokens = cf.forecast_num_tokens

        coords = self.get_coords(token_infos, num_tokens, token_sizes)
        lats, lons = self.reconstruct_geo_coords(token_infos, num_tokens, token_sizes)
        dates_t = self.extract_dates(token_infos, token_sizes.time)

        sources_out = self.process_input(sources)
        preds_out = self.process_predicted()

        # generate time range
        dates_sources, dates_targets = self.generate_time_range()

        levels = np.array(cf.fields[0][2])
        lats = [90.0 - lat for lat in lats]

        data = {
            OutputType.source: (sources_out, coords_sources),
            OutputType.pred: (preds_out, coords_targets),
        }

        self.store_batch(batch_idx, data)

    def process_input(self, sources, fields, dates_t, lats, lons):
        sources_out = []
        for field_idx, field_info in enumerate(fields):
            # reshape from tokens to contiguous physical field
            num_levels = len(field_info[2])
            vertical_levels = field_info[2]
            source = utils.detok(sources[field_idx].cpu().detach().numpy())

            # TODO: check that geo-coords match to general ones that have been pre-determined
            for date, lat, lon in zip(dates_t, lats, lons):
                for vertical_idx, _ in enumerate(vertical_levels):
                    denormalizer = self.model.normalizer(field_idx, vertical_idx)
                    date, coords = dates_t[bidx], [lats[bidx], lons[bidx]]
                    source[bidx, vertical_idx] = denormalizer.denormalize(
                        date.year, date.month, source[bidx, vertical_idx], coords
                    )
            # append
            sources_out.append([field_info[0], source])

        return sources_out

    def process_predicted(self):
        for fidx, fn in enumerate(self.config.fields_prediction):
            field_info = self.config.fields[self.fields_prediction_idx[fidx]]
            num_levels = len(field_info[2])
            # predictions
            pred = log_preds[fidx][0].cpu().detach().numpy()
            pred = utils.detok(
                pred.reshape(
                    [
                        num_levels,
                        -1,
                        forecast_num_tokens,
                        *field_info[3][1:],
                        *field_info[4],
                    ]
                ).swapaxes(0, 1)
            )

            # denormalize
            for vidx, vl in enumerate(field_info[2]):
                denormalize = self.model.normalizer(
                    self.fields_prediction_idx[fidx], vidx
                ).denormalize
                for bidx in range(token_infos[fidx].shape[0]):
                    date, coords = dates_t[bidx], [lats[bidx], lons[bidx]]
                    pred[bidx, vidx] = denormalize(
                        date.year, date.month, pred[bidx, vidx], coords
                    )

            # append
            preds_out.append([fn[0], pred])

    def denormalize(self, data, coords):
        dates_t, lats, lons = coords
        for vertical_idx, _ in enumerate(vertical_levels):
            denormalizer = self.model.normalizer(field_idx, vertical_idx)
            for token_idx, vertical_idx in it.product():
                date, coords = dates_t[bidx], [lats[bidx], lons[bidx]]
                data[bidx, vertical_idx] = self.denormalize(
                    denormalizer, data[bidx, vertical_idx], coord
                )

        return data

    @staticmethod
    def denormalize(denormalizer, data, coords):
        time_coord, spatial_coords = coords[0], coords[1:]
        return denormalizer.denormalize(
            time_coord.year, time_coord.month, data, spatial_coords
        )

    def generate_time_range(self, dates_t):
        dates_sources, dates_targets = [], []
        for bidx in range(source.shape[0]):
            r = pd.date_range(start=dates_t[bidx], periods=source.shape[2], freq="h")
            dates_sources.append(r.to_pydatetime().astype("datetime64[s]"))
