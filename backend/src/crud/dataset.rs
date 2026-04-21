use sqlx::PgPool;
use tokio::fs;

pub async fn cleanup_stale_datasets(db: &PgPool) {
    println!("Running stale dataset cleanup...");

    let datasets = match sqlx::query!("SELECT dataset_id, file_path FROM datasets")
        .fetch_all(db)
        .await
    {
        Ok(records) => records,
        Err(e) => {
            eprintln!("Failed to fetch datasets for cleanup: {}", e);
            return;
        }
    };

    let mut deleted_count = 0;

    for dataset in datasets {
        let exists = fs::try_exists(&dataset.file_path).await.unwrap_or(false);

        if !exists {
            let delete_result = sqlx::query!(
                "DELETE FROM datasets WHERE dataset_id = $1",
                dataset.dataset_id
            )
            .execute(db)
            .await;

            if let Err(e) = delete_result {
                eprintln!(
                    "Failed to delete stale dataset {}: {}",
                    dataset.dataset_id, e
                );
            } else {
                deleted_count += 1;
            }
        }
    }

    if deleted_count > 0 {
        println!(
            "Cleanup finished: Removed {} stale datasets.",
            deleted_count
        );
    } else {
        println!("Cleanup finished: No stale datasets found.");
    }
}
