package handler

import (
	"errors"
	"log/slog"
	"net/http"

	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/model"
	"github.com/autowarefoundation/auto_e2e/tools/datamodelconsole/api/internal/service"
)

// OccupancyModels handles
// GET /datasets/{name}/shards/{shard}/semantic-occupancy-models.
func (h *OverlayHandler) OccupancyModels(
	w http.ResponseWriter,
	r *http.Request,
) {
	dataset, shard, version, ok := h.shardRequest(w, r)
	if !ok {
		return
	}
	models, resolvedVersion, err := h.s3.ListSemanticOccupancyModels(
		r.Context(),
		dataset,
		version,
		shard,
	)
	if err != nil {
		if errors.Is(err, service.ErrNotFound) {
			writeError(
				w,
				http.StatusNotFound,
				model.CodeNotFound,
				"published shard not found",
			)
			return
		}
		slog.Error(
			"list semantic occupancy models",
			"dataset", dataset,
			"version", version,
			"shard", shard,
			"error", err,
		)
		writeError(
			w,
			http.StatusBadGateway,
			model.CodeUnavailable,
			"semantic occupancy catalog unavailable",
		)
		return
	}
	if models == nil {
		models = []model.SemanticOccupancyModel{}
	}
	writeJSON(w, http.StatusOK, model.SemanticOccupancyModelsResponse{
		Dataset: dataset,
		Version: resolvedVersion,
		Shard:   shard,
		Models:  models,
	})
}
